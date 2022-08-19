# Model Predicted Data
This folder contains our Woman and Migrant datasets with labels predicted by our ensemble.
 * 🗃 [`Woman` dataset](Frau-Predictions.json)
 * 🗃 [`Migrant` dataset](Migrant-Predictions.json)

Each json file contains a list of sentences with the following keys:

| Key           | Value  | Description                                                             |
| --------------|--------|-------------------------------------------------------------------------|
| `id`          | string | distinct id of the sentence                                             |
| `era`         | string | `rt` for Reichstag _or_ `bt` for Bundestag                              |
| `type`        | string | only for Reichstag data, see table above                                |
| `period`      | number | election period                                                         |
| `no`          | number | number of the sitting                                                   |
| `line`        | number | line number of the sentence                                             |
| `year`        | number | year of the sitting                                                     |
| `month`       | number | month of the sitting                                                    |
| `day`         | number | day of the sitting                                                      |
| `category`    | string | `Frau` for woman _or_ `Migrant` for migrant                             |
| `keyword`     | string | the keyword contained in the target sentence                            |
| `predictions` | array  | array of predictions of individual models                               |
| `label`       | string | `0` for solidarity _or_ `1` for anti-solidarity _or_ `2` for ambivalent |
