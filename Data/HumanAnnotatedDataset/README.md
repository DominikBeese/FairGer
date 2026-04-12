# Human Annotated Data

This folder contains the human-annotated [`Frau`](./HumanAnnotatedDatasetInputs_Frau.json) and [`Migrant`](./HumanAnnotatedDatasetInputs_Migrant.json) datasets used in the study.

For each target group, there are two files:

- an **input file**, which contains the instance to be annotated together with its metadata
- a **label file**, which contains the final annotation label for each instance id

Files and folders:

- 📄 [`HumanAnnotatedDatasetInputs_Frau.json`](./HumanAnnotatedDatasetInputs_Frau.json): input instances for the woman dataset
- 📄 [`HumanAnnotatedDatasetInputs_Migrant.json`](./HumanAnnotatedDatasetInputs_Migrant.json): input instances for the migrant dataset
- 🔒 [`labels_Frau.json.gpg`](./labels_Frau.json.gpg): encrypted labels for the woman dataset; see [Decrypting the label files](#decrypting-the-label-files)
- 🔒 [`labels_Migrant.json.gpg`](./labels_Migrant.json.gpg): encrypted labels for the migrant dataset; see [Decrypting the label files](#decrypting-the-label-files)
- 📄 [`LICENSE`](./LICENSE)

Each input json file contains a list of sentences with the following keys:

| Key               | Value  | Description                                 |
|-------------------|--------|---------------------------------------------|
| `id`              | string | distinct id of the instance                 |
| `era`             | string | `rt` for Reichstag _or_ `bt` for Bundestag  |
| `type`            | string | only for Reichstag data                     |
| `period`          | number | election period                             |
| `no`              | number | number of the sitting                       |
| `line`            | number | line number of the sentence                 |
| `year`            | number | year of the sitting                         |
| `month`           | number | month of the sitting                        |
| `day`             | number | day of the sitting                          |
| `category`        | string | `Frau` for woman _or_ `Migrant` for migrant |
| `keyword`         | string | the keyword contained in the target sentence |
| `prev_sents`      | array  | array of the preceding three sentences      |
| `sent`            | string | target sentence                             |
| `next_sents`      | array  | array of the following three sentences      |
| `consensus_level` | string | `curated`, `majority`, or `single`          |

Each label JSON file contains a list of objects with the following keys:

| Key     | Value  | Description                                          |
|---------|--------|------------------------------------------------------|
| `id`    | string | distinct id of the instance                          |
| `label` | string | final fine-grained annotation label for the instance |

The `label` field uses the following values:

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

## Decrypting the label files

The label files are distributed in encrypted form (`.gpg`).
To request access to the decryption password, please contact [aida.kostikova@uni-bielefeld.de](mailto:aida.kostikova@uni-bielefeld.de) or [lovegoodaida@gmail.com](mailto:lovegoodaida@gmail.com). The password is shared separately to reduce benchmark contamination risk. We aim to respond to password requests promptly.
After obtaining the password, you can decrypt the label files with:

```bash
gpg -o labels_Frau.json -d labels_Frau.json.gpg
gpg -o labels_Migrant.json -d labels_Migrant.json.gpg
```

## Benchmark protection

Please do not:
- redistribute decrypted label files publicly
- use protected labels for model training
- upload protected benchmark content to closed APIs without training-exclusion guarantees

This release follows the benchmark-protection rationale discussed in [Jacovi et al. (2023)](https://aclanthology.org/2023.emnlp-main.308/).

## License

The original contributions of the dataset authors, such as labels and annotations, are licensed under the Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0).

This license does not apply to underlying third-party materials or official-source texts, including Bundestag/Reichstag source texts, which remain subject to their own legal status and source-attribution requirements.
