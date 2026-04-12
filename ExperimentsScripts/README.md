# Experiment Scripts

This folder contains the scripts used to run the experiments reported in the project.

The subfolders are organized by model family or experimental setup:

- 📂 [`bert/`](./bert/): scripts for the hierarchical BERT baseline
- 📂 [`gpt-3.5_finetuning/`](./gpt-3.5_finetuning/): scripts for GPT-3.5 fine-tuning experiments
- 📂 [`gpt-inference-only/`](./gpt-inference-only/): scripts for inference-only experiments with GPT models
- 📂 [`llama-3-70b/`](./llama-3-70b/): scripts for inference-only experiments with Llama-3-70B
- 📄 [`ExtractLabel.py`](./ExtractLabel.py): helper script for extracting
- 📄 [`prompts.json`](./prompts.json): prompt templates used for prompting-based experiments

## Utility scripts

### [`ExtractLabel.py`](./ExtractLabel.py)

This script extracts final labels from raw model outputs and converts them to the label scheme used in this project.

It supports the following models:

- `GPT4`
- `GPT3.5`
- `Llama3`

The script:

- reads a prediction file in `.json` or `.csv` format
- looks for model response columns such as `model_response_GPT4`, `model_response_GPT3.5`, or `model_response_Llama3`
- extracts the label from generated text
- maps long-form labels such as `GROUP-BASED SOLIDARITY` to project labels such as `s.group-based` (see the [Label scheme](#label-scheme))
- writes the extracted label back into the same file

For `Llama3`, the script combines a high-level response and a subtype response.

## Label scheme

The experiments use the following labels:

| Label               | Description                     |
|---------------------|---------------------------------|
| `s.group-based`     | solidarity, group-based         |
| `s.exchange-based`  | solidarity, exchange-based      |
| `s.compassionate`   | solidarity, compassionate       |
| `s.empathic`        | solidarity, empathic            |
| `s.none`            | solidarity, no subtype          |
| `as.group-based`    | anti-solidarity, group-based    |
| `as.exchange-based` | anti-solidarity, exchange-based |
| `as.compassionate`  | anti-solidarity, compassionate  |
| `as.empathic`       | anti-solidarity, empathic       |
| `as.none`           | anti-solidarity, no subtype     |
| `mixed.none`        | mixed stance                    |
| `none.none`         | none                            |