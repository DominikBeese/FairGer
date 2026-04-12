import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val, torch.Tensor):
                item[key] = val[idx]
            else:
                item[key] = torch.tensor(val[idx])

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(model_checkpoint_path, tokenizer_name):
    if not os.path.isdir(model_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint_path,
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def load_id2label(task_dir):
    path = os.path.join(task_dir, "id2label.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing id2label.json in: {task_dir}")

    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def load_task_config(task_dir):
    config_path = os.path.join(task_dir, "config.json")
    if os.path.exists(config_path):
        return load_json(config_path)
    return {}


def preprocess_texts(texts, tokenizer, max_length=512):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def build_input_texts(df, encoding="target_only", add_category=False):
    if encoding == "target_only" and not add_category:
        texts = df["Middle"].astype(str).tolist()

    elif encoding == "target_only" and add_category:
        texts = (
            df["Category"].astype(str) + " [SEP] " + df["Middle"].astype(str)
        ).tolist()

    elif encoding == "target+context" and not add_category:
        texts = df.apply(
            lambda x: " [SEP] ".join(
                [str(x["Previous"]), str(x["Middle"]), str(x["Next"])]
            ),
            axis=1,
        ).tolist()

    elif encoding == "target+context" and add_category:
        texts = df.apply(
            lambda x: str(x["Category"]) + " [SEP] " + " [SEP] ".join(
                [str(x["Previous"]), str(x["Middle"]), str(x["Next"])]
            ),
            axis=1,
        ).tolist()

    else:
        raise ValueError(f"Unsupported encoding setting: {encoding}")

    return texts


def get_predictions(model, dataset):
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset)
    logits = predictions.predictions

    if isinstance(logits, torch.Tensor):
        return torch.argmax(logits, axis=-1).cpu().numpy()

    return np.argmax(logits, axis=-1)


def predict_with_task_model(df, model, tokenizer, config, id2label):
    encoding = config.get("encoding", "target_only")
    add_category = config.get("add_category", False)
    max_length = config.get("max_length", 512)

    texts = build_input_texts(
        df,
        encoding=encoding,
        add_category=add_category,
    )

    encodings = preprocess_texts(texts, tokenizer, max_length=max_length)
    dataset = TextDataset(encodings)
    pred_ids = get_predictions(model, dataset)
    pred_labels = [id2label[int(pred_id)] for pred_id in pred_ids]

    return pred_ids, pred_labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--major_task_dir", type=str, required=True)
    parser.add_argument("--s_task_dir", type=str, required=True)
    parser.add_argument("--as_task_dir", type=str, required=True)

    parser.add_argument("--major_checkpoint", type=str, required=True)
    parser.add_argument("--s_checkpoint", type=str, required=True)
    parser.add_argument("--as_checkpoint", type=str, required=True)

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--category_filter", type=str, default=None)
    parser.add_argument("--input_delimiter", type=str, default=";")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-german-cased")

    args = parser.parse_args()

    major_model, major_tokenizer = load_model_and_tokenizer(
        args.major_checkpoint,
        tokenizer_name=args.tokenizer_name,
    )
    s_model, s_tokenizer = load_model_and_tokenizer(
        args.s_checkpoint,
        tokenizer_name=args.tokenizer_name,
    )
    as_model, as_tokenizer = load_model_and_tokenizer(
        args.as_checkpoint,
        tokenizer_name=args.tokenizer_name,
    )

    major_id2label = load_id2label(args.major_task_dir)
    s_id2label = load_id2label(args.s_task_dir)
    as_id2label = load_id2label(args.as_task_dir)

    major_config = load_task_config(args.major_task_dir)
    s_config = load_task_config(args.s_task_dir)
    as_config = load_task_config(args.as_task_dir)

    df = pd.read_csv(args.input_csv, delimiter=args.input_delimiter)

    required_columns = ["Previous", "Middle", "Next", "Category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if args.category_filter is not None:
        df = df[
            df["Category"].astype(str).str.contains(args.category_filter, case=False, na=False)
        ].copy()

    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} rows for inference.")

    _, major_pred_labels = predict_with_task_model(
        df,
        major_model,
        major_tokenizer,
        major_config,
        major_id2label,
    )

    df["High_Level_Category"] = major_pred_labels
    df["Predicted_Label"] = np.nan

    s_mask = df["High_Level_Category"] == "s"
    as_mask = df["High_Level_Category"] == "as"
    mixed_mask = df["High_Level_Category"] == "mixed"
    none_mask = df["High_Level_Category"] == "none"

    if s_mask.any():
        _, s_pred_labels = predict_with_task_model(
            df.loc[s_mask].copy(),
            s_model,
            s_tokenizer,
            s_config,
            s_id2label,
        )
        df.loc[s_mask, "Predicted_Label"] = s_pred_labels

    if as_mask.any():
        _, as_pred_labels = predict_with_task_model(
            df.loc[as_mask].copy(),
            as_model,
            as_tokenizer,
            as_config,
            as_id2label,
        )
        df.loc[as_mask, "Predicted_Label"] = as_pred_labels

    df.loc[mixed_mask, "Predicted_Label"] = "mixed.none"

    df.loc[none_mask, "Predicted_Label"] = "none.none"

    print("\nPrediction preview:")
    print(df[["High_Level_Category", "Predicted_Label"]].head(20))

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()