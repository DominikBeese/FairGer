# Llama-3-70B Inference

[`llama-3-70b.py`](./llama-3-70b.py) is used for prompting-based inference with `meta-llama/Meta-Llama-3-70B-Instruct`.

### Configuration

The script is configured through constants at the top of the file:

- `MODEL_ID`: Hugging Face model ID
- `HF_TOKEN_ENV`: name of the environment variable containing the Hugging Face token
- `INPUT_FILE`: input dataset file (`.json` or `.csv`)
- `OUTPUT_FILE`: output file (`.json` or `.csv`)
- `PROMPTS_PATH`: path to `prompts.json`
- `MAIN_PROMPT_KEY`: prompt key for the high-level classification step
- `SOLIDARITY_SUBTYPE_PROMPT_KEY`: prompt key for the solidarity subtype step
- `ANTISOLIDARITY_SUBTYPE_PROMPT_KEY`: prompt key for the anti-solidarity subtype step
- `CATEGORY_FILTER`: category to process, for example `Migrant`; set to `None` to process all rows

### Input formats

The script supports two input formats:

- **CSV**, using the columns:
  - `Previous`
  - `Middle`
  - `Next`
  - `Category`

- **JSON**, using the keys:
  - `prev_sents`
  - `sent`
  - `next_sents`
  - `category`

For each instance, the prompt input text is built from concatenating previous, middle and next sentences.

### Processing behavior

The script uses a two-step inference setup:

1. it first predicts the high-level label (`SOLIDARITY`, `ANTI-SOLIDARITY`, `MIXED`, or `NONE`)
2. if the high-level label is `SOLIDARITY`, it runs a second prompt for the solidarity subtype
3. if the high-level label is `ANTI-SOLIDARITY`, it runs a second prompt for the anti-solidarity subtype
Rows classified as `MIXED` or `NONE` do not receive a subtype prompt.

- Only rows matching `CATEGORY_FILTER` (Frau or Migrant) are processed.
- The script can resume from an existing output file. Rows that already contain the required response fields are skipped.
- Progress is saved after each processed row.

### Output

The script writes raw model responses into two columns:

- `model_response_Llama3`: high-level model response
- `model_response_Llama3_subtype`: subtype model response

Final project labels are extracted from these columns with [`../ExtractLabel.py`](../ExtractLabel.py).

### Requirements

Tested with Python 3.10.19 and the package versions listed in `requirements.txt`. 

The script requires access to `meta-llama/Meta-Llama-3-70B-Instruct` on Hugging Face and a Hugging Face access token in the environment:

```bash
export HF_TOKEN=...
```