import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


MODEL_NAME = "gpt-4-1106-preview"  # or gpt-3.5-turbo-0125
TARGET_CATEGORY = "Migrant"
PROMPT_KEY = "Migrant_zeroshot"
PROMPTS_PATH = "prompts.json"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_response_column_name(model_name: str) -> str:
    if model_name == "gpt-4-1106-preview":
        return "model_response_GPT4"
    if model_name == "gpt-3.5-turbo-0125":
        return "model_response_GPT3.5"
    raise ValueError(
        f"Unsupported MODEL_NAME '{model_name}'. "
        "Expected 'gpt-4-1106-preview' or 'gpt-3.5-turbo-0125'."
    )


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def api_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def normalize_context_field(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def build_input_text(data_point: dict) -> str:
    parts = [
        data_point["context1"].strip(),
        data_point["input"].strip(),
        data_point["context2"].strip(),
    ]
    return " ".join(part for part in parts if part)


def build_prompt(prompt_template: str, data_point: dict) -> str:
    input_text = build_input_text(data_point)
    return f"{prompt_template}\n\n### Input Text:\n{input_text}"


def load_input_data(file_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, delimiter=";")
        return df, "csv"

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df, "json"

    raise ValueError(f"Unsupported input format: {suffix}")


def save_output_data(df: pd.DataFrame, file_path: str, file_type: str) -> None:
    path = Path(file_path)

    if file_type == "csv":
        df.to_csv(path, index=False, sep=";")
        return

    if file_type == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        return

    raise ValueError(f"Unsupported output format: {file_type}")


def row_matches_category(row: pd.Series, file_type: str, target_category: str) -> bool:
    if file_type == "csv":
        row_category = str(row.get("Category", "")).strip().lower()
        return row_category == target_category.lower()

    if file_type == "json":
        row_category = str(row.get("category", "")).strip().lower()
        return row_category == target_category.lower()

    return False


def build_data_point(row: pd.Series, file_type: str) -> dict:
    if file_type == "csv":
        return {
            "context1": normalize_context_field(row.get("Previous", "")),
            "input": normalize_context_field(row.get("Middle", "")),
            "context2": normalize_context_field(row.get("Next", "")),
            "category": str(row.get("Category", "")),
        }

    if file_type == "json":
        return {
            "context1": normalize_context_field(row.get("prev_sents", "")),
            "input": normalize_context_field(row.get("sent", "")),
            "context2": normalize_context_field(row.get("next_sents", "")),
            "category": str(row.get("category", "")),
        }

    raise ValueError(f"Unsupported file type: {file_type}")


def process_and_save_dataset(
    input_file_path: str,
    target_category: str,
    output_file_path: str,
    prompts_path: str,
    prompt_key: str,
) -> None:
    df, file_type = load_input_data(input_file_path)

    prompts = load_prompts(prompts_path)
    if prompt_key not in prompts:
        raise KeyError(f"Prompt key '{prompt_key}' not found in {prompts_path}")

    prompt_template = prompts[prompt_key]
    response_col = get_response_column_name(MODEL_NAME)

    if response_col not in df.columns:
        df[response_col] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        existing_response = row.get(response_col)
        needs_processing = pd.isnull(existing_response) or str(existing_response).strip() == ""

        if needs_processing and row_matches_category(row, file_type, target_category):
            try:
                data_point = build_data_point(row, file_type)
                prompt = build_prompt(prompt_template, data_point)
                model_response = api_call(prompt)
                df.at[index, response_col] = model_response

            except Exception as e:
                print(f"Error processing row {index}: {e}")

        if index > 0 and index % 10 == 0:
            save_output_data(df, output_file_path, file_type)

    save_output_data(df, output_file_path, file_type)
    print(f"Final data saved to {output_file_path}.")


if __name__ == "__main__":
    process_and_save_dataset(
        input_file_path="Migrant18k.json",
        target_category=TARGET_CATEGORY,
        output_file_path="Migrant18kGPT4_predicted.json",
        prompts_path=PROMPTS_PATH,
        prompt_key=PROMPT_KEY,
    )