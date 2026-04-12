# Model Predicted Data
This folder contains a sampled subset of the `Migrant` dataset with labels predicted by GPT-4. It includes 18,300 migrant instances, which correspond to a proportional sample of the full migrant dataset.

Each json file contains a list of sentences with the following keys:

| Key                    | Value  | Description                                              |
|------------------------|--------|----------------------------------------------------------|
| `id`                   | string | distinct id of the sentence                              |
| `era`                  | string | `rt` for Reichstag _or_ `bt` for Bundestag               |
| `type`                 | string | only for Reichstag data, see table above                 |
| `period`               | number | election period                                          |
| `no`                   | number | number of the sitting                                    |
| `line`                 | number | line number of the sentence                              |
| `year`                 | number | year of the sitting                                      |
| `month`                | number | month of the sitting                                     |
| `day`                  | number | day of the sitting                                       |
| `category`             | string | `Migrant` for migrant                                    |
| `keyword`              | string | the keyword contained in the target sentence             |
| `party`                | string | extracted party, where available                         |
| `prev_sents`           | array  | array of the preceding three sentences                   |
| `sent`                 | string | target sentence                                          |
| `next_sents`           | array  | array of the following three sentences                   |
| `model_response_GPT4`  | string | raw GPT-4 response for the instance                      |
| `extracted_label_GPT4` | string | final project label extracted from `model_response_GPT4` |
