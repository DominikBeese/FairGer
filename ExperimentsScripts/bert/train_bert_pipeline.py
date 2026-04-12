from argparse import ArgumentParser
from collections import Counter
import json
import math
import os
import pprint
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    logging,
)

logging.set_verbosity_error()
os.environ["WANDB_DISABLED"] = "true"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, dropout=0.1, device="cuda", weight=None, num_labels=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    if hasattr(model, "classifier") and hasattr(model.classifier, "dropout"):
        model.classifier.dropout = nn.Dropout(p=dropout)

    if weight:
        model.load_state_dict(torch.load(weight, map_location=device))

    model.eval()
    model.to(device)
    return tokenizer, model, config


def oversample(df, on="label"):
    """
    Randomly sample minority classes until all classes match the majority size.
    """
    max_size = df[on].value_counts().max()
    chunks = [df]

    print(f"Oversampling on column: {on}")
    for _, group in df.groupby(on):
        if len(group) < max_size:
            chunks.append(
                group.sample(max_size - len(group), replace=True, random_state=42)
            )

    df_new = pd.concat(chunks)
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_new


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(self.data["input_ids"])


def freeze_first_n_layers(model, n=10, model_type="bert"):
    """
    Freeze the first n layers of BERT/ROBERTA during training.
    """
    if model_type == "bert":
        frozen_layers = [f"{model_type}.encoder.layer.{i}." for i in range(n)]
    elif model_type == "roberta":
        frozen_layers = [f"{model_type}.encoder.layer.{i}." for i in range(n)]
    elif model_type == "llama":
        frozen_layers = [f"model.layers.{i}." for i in range(n)]
    else:
        frozen_layers = []

    print("Layers to be frozen:")
    print(frozen_layers)

    for name, param in model.named_parameters():
        param.requires_grad = True
        for layer in frozen_layers:
            if name.startswith(layer):
                param.requires_grad = False
                break

    print("Layers to be updated:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def preprocess(
    df,
    tokenizer,
    encoding="target_only",
    model_type="bert",
    add_category=False,
    max_length=512,
    label_col=None,
    label2id=None,
):
    model_type = "roberta" if "roberta" in model_type else "bert"

    if model_type == "bert":
        if encoding == "target_only" and add_category is False:
            texts = df["Middle"].tolist()
        elif encoding == "target_only" and add_category is True:
            texts = df.apply(
                lambda x: x["Category"] + " [SEP] " + x["Middle"], axis=1
            ).tolist()
        elif encoding == "target+context" and add_category is False:
            texts = df.apply(
                lambda x: " [SEP] ".join([x["Previous"], x["Middle"], x["Next"]]),
                axis=1,
            ).tolist()
        elif encoding == "target+context" and add_category is True:
            texts = df.apply(
                lambda x: x["Category"] + " [SEP] " + " [SEP] ".join(
                    [x["Previous"], x["Middle"], x["Next"]]
                ),
                axis=1,
            ).tolist()
        else:
            raise NotImplementedError("Unsupported encoding setting.")
    else:
        raise NotImplementedError("Only bert/roberta preprocessing is implemented.")

    data = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if label_col is not None:
        if label2id is None:
            label2id = {
                label: i for i, label in enumerate(sorted(df[label_col].dropna().unique()))
            }
        data["labels"] = torch.tensor([label2id[label] for label in df[label_col].tolist()])

    return data, label2id


def compute_metrics(eval_pred, label_id_map):
    predictions, labels = eval_pred
    predictions = softmax(predictions, axis=1)
    predictions = np.argmax(predictions, axis=-1)

    id_label_map = {v: k for k, v in label_id_map.items()}
    predicted_labels = [id_label_map[label_id] for label_id in predictions]
    true_labels = [id_label_map[label_id] for label_id in labels]

    print()
    print(classification_report(y_true=true_labels, y_pred=predicted_labels, digits=6))

    report = classification_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    return report["macro avg"]


def get_optimizer_grouped_parameters(
    model,
    model_type,
    learning_rate,
    weight_decay,
    layerwise_learning_rate_decay,
):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "classifier" in n or "pooler" in n
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]

    encoder_root = getattr(model, model_type)
    layers = [encoder_root.embeddings] + list(encoder_root.encoder.layer)
    layers.reverse()

    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters.extend(
            [
                {
                    "params": [
                        p for n, p in layer.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in layer.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        )

    return optimizer_grouped_parameters


def prepare_split(df, task):
    df = df.copy()

    if task == "major":
        target_col = "major"

    elif task == "s":
        df = df[df["major"] == "s"]
        df = df[df["label"].astype(str).str.startswith("s.")]
        df = df[~df["label"].astype(str).str.contains("s.none", case=False, na=False)]
        target_col = "label"

    elif task == "as":
        df = df[df["major"] == "as"]
        df = df[df["label"].astype(str).str.startswith("as.")]
        df = df[~df["label"].astype(str).str.contains("as.none", case=False, na=False)]
        target_col = "label"

    else:
        raise ValueError(f"Unknown task: {task}")

    df = df[df[target_col].notna()].reset_index(drop=True)
    return df, target_col


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--task", type=str, choices=["major", "s", "as"], required=True)
    parser.add_argument("--id", type=int, default=0, help="A unique id assigned to this training experiment")
    parser.add_argument("--data_dir", type=str, default="data/training_splits")
    parser.add_argument("--model_type", type=str, default="bert-base-german-cased")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--encoding", type=str, choices=["target_only", "target+context"], default="target_only",)
    parser.add_argument("--add_category", action="store_true")
    parser.add_argument("--oversample", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20, help="Maximal number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--freeze_first_n_layers", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--layerwise_learning_rate_decay", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--not_save_checkpoint", action="store_true")
    parser.add_argument("--best_metric", type=str, default="eval_f1-score")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    global_args = parser.parse_args()
    pprint.pprint(vars(global_args), indent=4)

    seed_everything(global_args.seed)

    out_dir = os.path.join(global_args.save_dir, global_args.task, str(global_args.id))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(global_args), f, indent=4)

    data_dir = global_args.data_dir

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), delimiter=";")
    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"), delimiter=";")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), delimiter=";")

    train_df, target_col = prepare_split(train_df, global_args.task)
    dev_df, _ = prepare_split(dev_df, global_args.task)
    test_df, _ = prepare_split(test_df, global_args.task)

    print()
    print(f"Task: {global_args.task}")
    print(f"Target column: {target_col}")
    print("Train distribution:", Counter(train_df[target_col]))
    print("Dev distribution:", Counter(dev_df[target_col]))
    print("Test distribution:", Counter(test_df[target_col]))

    if global_args.oversample:
        train_df = oversample(train_df, on=target_col)
        print("Train distribution after oversampling:", Counter(train_df[target_col]))

    num_labels = train_df[target_col].nunique()

    tokenizer, model, _ = load_model(
        global_args.model_type,
        dropout=global_args.dropout,
        device=global_args.device,
        weight=global_args.weight,
        num_labels=num_labels,
    )

    model_type = "roberta" if "roberta" in global_args.model_type else "bert"
    freeze_first_n_layers(model, n=global_args.freeze_first_n_layers, model_type=model_type)

    train_data, train_label2id = preprocess(
        train_df,
        tokenizer,
        global_args.encoding,
        model_type,
        global_args.add_category,
        global_args.max_length,
        target_col,
        label2id=None,
    )

    id2label = {v: k for k, v in train_label2id.items()}

    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(train_label2id, f, indent=4, ensure_ascii=False)

    with open(os.path.join(out_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=4, ensure_ascii=False)

    model.config.label2id = train_label2id
    model.config.id2label = id2label

    dev_data, _ = preprocess(
        dev_df,
        tokenizer,
        global_args.encoding,
        model_type,
        global_args.add_category,
        global_args.max_length,
        target_col,
        label2id=train_label2id,
    )
    test_data, _ = preprocess(
        test_df,
        tokenizer,
        global_args.encoding,
        model_type,
        global_args.add_category,
        global_args.max_length,
        target_col,
        label2id=train_label2id,
    )

    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)

    steps_per_epoch = math.ceil(len(train_df) / global_args.batch_size)
    total_steps = steps_per_epoch * global_args.epochs
    actual_warmup_steps = (
        math.ceil(total_steps * global_args.warmup_ratio)
        if global_args.warmup_steps == -1
        else global_args.warmup_steps
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=global_args.epochs,
        per_device_train_batch_size=global_args.batch_size,
        per_device_eval_batch_size=128,
        warmup_steps=actual_warmup_steps,
        learning_rate=global_args.learning_rate,
        weight_decay=global_args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        seed=global_args.seed,
        metric_for_best_model=global_args.best_metric,
        save_total_limit=1,
        fp16=True if global_args.device == "cuda" else False,
        gradient_accumulation_steps=global_args.gradient_accumulation_steps,
        load_best_model_at_end=True,
    )

    grouped_optimizer_params = get_optimizer_grouped_parameters(
        model,
        model_type,
        global_args.learning_rate,
        global_args.weight_decay,
        global_args.layerwise_learning_rate_decay,
    )

    optimizer = AdamW(
        grouped_optimizer_params,
        lr=global_args.learning_rate,
        eps=1e-8,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=actual_warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, train_label2id),
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    print("\nDev evaluation:")
    trainer.evaluate(eval_dataset=dev_dataset)

    print("\nTest evaluation:")
    trainer.evaluate(eval_dataset=test_dataset)

    if global_args.not_save_checkpoint:
        shutil.rmtree(out_dir, ignore_errors=True)
        print("Deleted checkpoint directory.")