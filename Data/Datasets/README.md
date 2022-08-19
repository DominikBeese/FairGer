# Datasets
This folder contains our Woman and Migrant datasets.
 * 🗃 `Woman` dataset ([Part 1](Frau1.json), [Part 2](Frau2.json))
 * 🗃 [`Migrant` dataset](Migrant.json)

Each json file contains a list of sentences with the following keys:

| Key          | Value  | Description                                  |
| -------------|--------|----------------------------------------------|
| `id`         | string | distinct id of the sentence                  |
| `era`        | string | `rt` for Reichstag _or_ `bt` for Bundestag   |
| `type`       | string | only for Reichstag data, see table above     |
| `period`     | number | election period                              |
| `no`         | number | number of the sitting                        |
| `line`       | number | line number of the sentence                  |
| `year`       | number | year of the sitting                          |
| `month`      | number | month of the sitting                         |
| `day`        | number | day of the sitting                           |
| `category`   | string | `Frau` for woman _or_ `Migrant` for migrant  |
| `keyword`    | string | the keyword contained in the target sentence |
| `prev_sents` | array  | array of the preceeding three sentences      |
| `sent`       | string | target sentence                              |
| `next_sents` | array  | array of the following three sentences       |
