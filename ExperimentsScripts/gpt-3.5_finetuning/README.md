# GPT-3.5 Fine-Tuning

This folder contains the data used for the GPT-3.5 fine-tuning experiments.

## Files

- [`train_frau_gpt_explanations.jsonl`](./train_frau_gpt_explanations.jsonl): fine-tuning data for women-related instances
- [`train_migrant_gpt_explanations.jsonl`](./train_migrant_gpt_explanations.jsonl): fine-tuning data for migrant-related instances

## Format

Each line in the JSONL files contains one object with a `messages` field:

- `system`: task instructions and label definitions
- `user`: German input text
- `assistant`: explanation (generated with GPT-4) and final label

Example structure:

```json
{
  "messages": [
    {"role": "system", "content": "... task instructions and label definitions ..."},
    {"role": "user", "content": "... input text ..."},
    {"role": "assistant", "content": "... explanation and final label ..."}
  ]
}
```