# Human Annotated Data
This folder contains our human annotated Woman and Migrant datasets with human annotations. It also contains the dataset of backtranslations, and a list of IDs of all 270 instances where disagreements were resolved either by discussion or majority vote ("Golden 270").
 * 🗃 [`Woman` dataset](Frau.json)
 * 🗃 [`Migrant` dataset](Migrant.json)
 * 🗃 [Backtranslation dataset](Backtranslations.json)
 * 🗃 [IDs of the "Golden 270"](Golden270.txt)

Each json file contains a list of sentences with the following keys:

| Key          | Value  | Description                                                             |
| -------------|--------|-------------------------------------------------------------------------|
| `id`         | string | distinct id of the sentence                                             |
| `era`        | string | `rt` for Reichstag _or_ `bt` for Bundestag                              |
| `type`       | string | only for Reichstag data, see table above                                |
| `period`     | number | election period                                                         |
| `no`         | number | number of the sitting                                                   |
| `line`       | number | line number of the sentence                                             |
| `year`       | number | year of the sitting                                                     |
| `month`      | number | month of the sitting                                                    |
| `day`        | number | day of the sitting                                                      |
| `category`   | string | `Frau` for woman _or_ `Migrant` for migrant                             |
| `keyword`    | string | the keyword contained in the target sentence                            |
| `prev_sents` | array  | array of the preceeding three sentences                                 |
| `sent`       | string | target sentence                                                         |
| `next_sents` | array  | array of the following three sentences                                  |
| `label`      | string | `0` for solidarity _or_ `1` for anti-solidarity _or_ `2` for ambivalent |
