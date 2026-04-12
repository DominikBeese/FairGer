from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

INPUT_DIR = Path("../../Data/Datasets")
OUTPUT_PATH = Path("../../Analysis/KeywordAnalysis/KeywordDistributionOverTime/DistributionOfKeywords.pdf")

DATASETS = ["Frau", "Migrant"]
DISPLAY_NAMES = {"Frau": "Woman", "Migrant": "Migrant"}

FIGSIZE = (5.4, 3.0)
KEYWORD_SHARE_THRESHOLD = 0.035


def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load dataset files and keep only required columns."""
    dfs = {
        name: pd.read_json(input_dir / f"{name}.json")
        for name in DATASETS
    }

    dfs = {
        name: df[["id", "year", "month", "day", "category", "keyword"]].copy()
        for name, df in dfs.items()
    }

    dfs["Frau"]["category"] = "Woman"
    return dfs


def prepare_keyword_distribution(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate keyword frequencies and collapse small shares into 'Other'."""
    dft = (
        df.groupby("keyword")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

    dft["percentage"] = dft["count"] / dft["count"].sum()

    drop_mask = dft["percentage"] < threshold
    dropped = dft[drop_mask].copy()
    kept = dft[~drop_mask].copy()

    kept["keyword"] = [f"${keyword}$" for keyword in kept["keyword"]]

    if not dropped.empty:
        kept = pd.concat(
            [
                kept,
                pd.DataFrame(
                    {
                        "keyword": ["Other"],
                        "count": [dropped["count"].sum()],
                        "percentage": [dropped["percentage"].sum()],
                    }
                ),
            ],
            ignore_index=True,
        )

    return kept


def plot_data(dfs: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot keyword distributions as two pie charts."""
    plt.figure(figsize=FIGSIZE)
    plt.subplots_adjust(left=0.04, right=0.86, top=0.82, bottom=0.10)

    for i, (dataset_name, df) in enumerate(dfs.items(), start=1):
        plt.subplot(1, 2, i)

        dft = prepare_keyword_distribution(df, KEYWORD_SHARE_THRESHOLD)

        colors = sns.color_palette("light:b_r", n_colors=len(dft) - 1)
        if "Other" in dft["keyword"].values:
            colors = colors + [(0.8, 0.8, 0.8)]

        plt.pie(
            x=dft["percentage"],
            labels=dft["keyword"],
            colors=colors,
            startangle=180,
            counterclock=False,
            pctdistance=0.8,
            autopct="%.0f%%",
            textprops={"size": 9},
        )

        plt.title(DISPLAY_NAMES.get(dataset_name, dataset_name))

    plt.suptitle("Distribution of Keywords", x=0.45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def main() -> None:
    dfs = load_data(INPUT_DIR)
    plot_data(dfs, OUTPUT_PATH)


if __name__ == "__main__":
    main()