## GPT inference script

The inference script runs prompting-based classification with OpenAI chat models on dataset files in `.json` or `.csv` format.

### Configuration

The script is configured through constants at the top of the file:

- `MODEL_NAME`: OpenAI model name, for example `gpt-4-1106-preview` or `gpt-3.5-turbo-0125`
- `TARGET_CATEGORY`: category to process, for example `Migrant`
- `PROMPT_KEY`: key used to select the prompt template from `prompts.json`
- `PROMPTS_PATH`: path to the prompt file

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

- Only rows matching `TARGET_CATEGORY` (Frau or Migrant) are processed.
- Rows that already contain a response in the target output column are skipped.
- Progress is saved every 10 rows.

The script writes model responses into a model-specific column:

- `model_response_GPT4` for `gpt-4-1106-preview`
- `model_response_GPT3.5` for `gpt-3.5-turbo-0125`

Final project labels are extracted from these `model_response_*` columns with [`ExtractLabel.py`](../ExtractLabel.py).

### Requirements

The script requires an OpenAI API key in the environment:

```bash
export OPENAI_API_KEY=...
```