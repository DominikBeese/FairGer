import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import BitsAndBytesConfig, pipeline


MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
HF_TOKEN_ENV = "HF_TOKEN"

INPUT_FILE = "INPUT_FILE.json"    # json or csv
OUTPUT_FILE = "OUTPUT_FILE.json"  # json or csv

PROMPTS_PATH = "prompts.json"
MAIN_PROMPT_KEY = "Migrant_fewshot"
SOLIDARITY_SUBTYPE_PROMPT_KEY = "Migrant_solidarity_zeroshot"
ANTISOLIDARITY_SUBTYPE_PROMPT_KEY = "Migrant_antisolidarity_zeroshot"

CATEGORY_FILTER = "Migrant"  # set to None to process all rows

MODEL_RESPONSE_COL = "model_response_Llama3"
MODEL_RESPONSE_SUBTYPE_COL = "model_response_Llama3_subtype"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.6
TOP_P = 0.9


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model():
    hf_token = os.getenv(HF_TOKEN_ENV)
    if not hf_token:
        raise EnvironmentError(f"{HF_TOKEN_ENV} environment variable is not set.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    return pipeline(
        "text-generation",
        model=MODEL_ID,
        token=hf_token,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        },
    )


def extract_high_level_label(response_text: str) -> str | None:
    if not isinstance(response_text, str):
        return None

    match = re.search(
        r"LABEL\s*[:\-]?\s*(ANTI-SOLIDARITY|SOLIDARITY|MIXED|NONE)\b",
        response_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().upper()

    upper_text = response_text.upper()
    for label in ["ANTI-SOLIDARITY", "SOLIDARITY", "MIXED", "NONE"]:
        if re.search(rf"\b{re.escape(label)}\b", upper_text):
            return label

    return None


def generate_response(text_generator, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]

    prompt = text_generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    eos_token_id = text_generator.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID is undefined. Check tokenizer initialization.")

    outputs = text_generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    generated_text = outputs[0]["generated_text"]
    if len(generated_text) > len(prompt):
        return generated_text[len(prompt):].strip()
    return ""


def normalize_sentences(field) -> list[str]:
    if isinstance(field, list):
        return [s.strip() for s in field if isinstance(s, str) and s.strip()]
    if isinstance(field, str):
        field = field.strip()
        return [field] if field else []
    return []


def build_full_text(row: pd.Series) -> str:
    prev_sents = normalize_sentences(row.get("prev_sents", ""))
    sent = str(row.get("sent", "")).strip()
    next_sents = normalize_sentences(row.get("next_sents", ""))

    if not sent:
        raise ValueError("Missing 'sent' field in row.")

    return " ".join(prev_sents + [sent] + next_sents)


def save_progress(df: pd.DataFrame, output_file: str, file_type: str) -> None:
    path = Path(output_file)

    if file_type == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        return

    if file_type == "csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {file_type}")


def load_input_data(file_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f)), "json"

    if suffix == ".csv":
        return pd.read_csv(path), "csv"

    raise ValueError(f"Unsupported input format: {suffix}")


def row_matches_category(row: pd.Series, target_category: str | None) -> bool:
    if not target_category:
        return True

    row_category = str(row.get("category", "")).strip().lower()
    return row_category == target_category.lower()


def already_processed(row: pd.Series) -> bool:
    high = row.get(MODEL_RESPONSE_COL)
    sub = row.get(MODEL_RESPONSE_SUBTYPE_COL)

    if not isinstance(high, str) or not high.strip():
        return False

    high_label = extract_high_level_label(high)

    if high_label in {"NONE", "MIXED"}:
        return True

    if high_label in {"SOLIDARITY", "ANTI-SOLIDARITY"}:
        return isinstance(sub, str) and bool(sub.strip())

    return False


def process_text(
    row: pd.Series,
    text_generator,
    prompts: dict,
    main_prompt_key: str,
    solidarity_subtype_prompt_key: str,
    antisolidarity_subtype_prompt_key: str,
) -> tuple[str, str | None]:
    if main_prompt_key not in prompts:
        raise KeyError(f"Prompt key '{main_prompt_key}' not found in {PROMPTS_PATH}")

    if solidarity_subtype_prompt_key not in prompts:
        raise KeyError(
            f"Prompt key '{solidarity_subtype_prompt_key}' not found in {PROMPTS_PATH}"
        )

    if antisolidarity_subtype_prompt_key not in prompts:
        raise KeyError(
            f"Prompt key '{antisolidarity_subtype_prompt_key}' not found in {PROMPTS_PATH}"
        )

    full_text = build_full_text(row)

    high_level_response = generate_response(
        text_generator=text_generator,
        instruction=prompts[main_prompt_key],
        text=full_text,
    )

    high_level_label = extract_high_level_label(high_level_response)
    subtype_response = None

    if high_level_label == "SOLIDARITY":
        subtype_response = generate_response(
            text_generator=text_generator,
            instruction=prompts[solidarity_subtype_prompt_key],
            text=full_text,
        )
    elif high_level_label == "ANTI-SOLIDARITY":
        subtype_response = generate_response(
            text_generator=text_generator,
            instruction=prompts[antisolidarity_subtype_prompt_key],
            text=full_text,
        )

    return high_level_response, subtype_response


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    prompts = load_prompts(PROMPTS_PATH)
    text_generator = build_model()

    if output_path.exists():
        df, file_type = load_input_data(str(output_path))
        print(f"Resuming from existing output: {output_path}")
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df, file_type = load_input_data(str(input_path))
        print(f"Starting from scratch using input: {input_path}")

    if MODEL_RESPONSE_COL not in df.columns:
        df[MODEL_RESPONSE_COL] = None

    if MODEL_RESPONSE_SUBTYPE_COL not in df.columns:
        df[MODEL_RESPONSE_SUBTYPE_COL] = None

    for index, row in df.iterrows():
        if not row_matches_category(row, CATEGORY_FILTER):
            continue

        if already_processed(row):
            continue

        try:
            high_response, subtype_response = process_text(
                row=row,
                text_generator=text_generator,
                prompts=prompts,
                main_prompt_key=MAIN_PROMPT_KEY,
                solidarity_subtype_prompt_key=SOLIDARITY_SUBTYPE_PROMPT_KEY,
                antisolidarity_subtype_prompt_key=ANTISOLIDARITY_SUBTYPE_PROMPT_KEY,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"Skipping index {index} due to CUDA OOM.")
            continue
        except Exception as e:
            print(f"Error at index {index}: {e}")
            continue

        df.at[index, MODEL_RESPONSE_COL] = high_response
        df.at[index, MODEL_RESPONSE_SUBTYPE_COL] = subtype_response

        save_progress(df, str(output_path), file_type)
        print(f"Saved progress after index {index}")

    print("Processing complete.")


if __name__ == "__main__":
    main()