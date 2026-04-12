# Hierarchical BERT Text Classification

This repository contains training and inference code for a hierarchical text classification pipeline based on German BERT (`bert-base-german-cased`).

The pipeline uses three classifiers:

- `major`: predicts the high-level category (`s`, `as`, `mixed`, `none`)
- `s`: predicts one of `s.group-based`, `s.exchange-based`, `s.compassionate`, or `s.empathic` for instances classified as `s`
- `as`: predicts one of `as.group-based`, `as.exchange-based`, `as.compassionate`, or `as.empathic` for instances classified as `as`

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10.

## Training

### Required columns:

- `Previous`: preceding sentences
- `Middle`: target sentence
- `Next`: following sentences
- `Category`: target group, for example `Frau` or `Migrant`
- `major`: high-level label (`s`, `as`, `mixed`, or `none`)
- `label`: full fine-grained label, for example `s.group-based`, `as.empathic`, `mixed.none`, or `none.none`

### Task Definitions

- `major`: predicts the `major` column
- `s`: is trained only on rows where `major == "s"` and predicts the `label` column
- `as`: is trained only on rows where `major == "as"` and predicts the `label` column

Run the following commands to train the three classifiers separately.
- 
Train the `major` classifier:

```bash
python train_bert_pipeline.py --task major --device cuda --batch_size 16 --data_dir data --encoding target+context --add_category --oversample --learning_rate 4e-4 --warmup_ratio 0.05 --freeze_first_n_layers 12 --layerwise_learning_rate_decay 1 --epochs 20 --save_dir outputs/checkpoints
```

Train the `s` classifier:

```bash
python train_bert_pipeline.py --task s --device cuda --batch_size 16 --data_dir data --encoding target+context --add_category --oversample --learning_rate 4e-4 --warmup_ratio 0.05 --freeze_first_n_layers 12 --layerwise_learning_rate_decay 1 --epochs 20 --save_dir outputs/checkpoints
```

Train the `as` classifier:

```bash
python train_bert_pipeline.py --task as --device cuda --batch_size 16 --data_dir data --encoding target+context --add_category --oversample --learning_rate 4e-4 --warmup_ratio 0.05 --freeze_first_n_layers 12 --layerwise_learning_rate_decay 1 --epochs 20 --save_dir outputs/checkpoints
```

## Outputs

When the training script is run, outputs are saved under:

```text
outputs/checkpoints/<task>/<id>/
```

These may include:

- `config.json`: training arguments for the run
- `label2id.json`: mapping from label names to numeric IDs
- `id2label.json`: reverse mapping from numeric IDs to label names
- model checkpoints

## Notes

- The script supports both `target_only` input encoding, which uses only the `Middle` sentence, as well as `target+context` input encoding, which uses `Previous`, `Middle`, and `Next`. With `--add_category`, the `Category` field is prepended to the input.

## Inference

Run hierarchical prediction on a CSV file:

```bash
python predict_bert_pipeline.py \
  --major_task_dir outputs/checkpoints/major/0 \
  --s_task_dir outputs/checkpoints/s/0 \
  --as_task_dir outputs/checkpoints/as/0 \
  --major_checkpoint outputs/checkpoints/major/0/<major_checkpoint_dir> \
  --s_checkpoint outputs/checkpoints/s/0/<s_checkpoint_dir> \
  --as_checkpoint outputs/checkpoints/as/0/<as_checkpoint_dir> \
  --input_csv data/test.csv \
  --output_csv predictions.csv
```

### Arguments

- `--major_task_dir`: path to the high-level model run directory. This directory should contain files such as `config.json` and `id2label.json`.
- `--s_task_dir`: path to the solidarity subtype model run directory.
- `--as_task_dir`: path to the anti-solidarity subtype model run directory.
- `--major_checkpoint`: path to the saved checkpoint for the high-level classifier.
- `--s_checkpoint`: path to the saved checkpoint for the solidarity subtype classifier.
- `--as_checkpoint`: path to the saved checkpoint for the anti-solidarity subtype classifier.
- `--input_csv`: path to the input CSV file used for inference.
- `--output_csv`: path to the output CSV file where predictions will be saved.