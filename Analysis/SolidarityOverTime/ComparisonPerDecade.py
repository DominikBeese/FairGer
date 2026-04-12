from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


sns.set_theme(style="whitegrid")

INPUT_PATH = Path("../../Data/ModelPredictedData/Migrant18kGPT4_predicted.json")
OUTPUT_PATH = Path("../../Analysis/SolidarityOverTime/ComparisonPerDecade-Migrant.pdf")

LABEL_MAP = {
    "s.group-based": "solidarity",
    "s.exchange-based": "solidarity",
    "s.compassionate": "solidarity",
    "s.empathic": "solidarity",
    "as.group-based": "anti-solidarity",
    "as.exchange-based": "anti-solidarity",
    "as.compassionate": "anti-solidarity",
    "as.empathic": "anti-solidarity",
    "mixed.none": "mixed",
    "none.none": "none",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file and keep only required columns."""
    df = pd.read_json(path)
    return df[["id", "year", "extracted_label_GPT4"]].copy()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map labels, aggregate by decade, and compute derived series."""
    df["high_level_label"] = df["extracted_label_GPT4"].map(LABEL_MAP)
    df = df[df["high_level_label"].notna()].copy()

    df["decade"] = (df["year"] // 10) * 10 + 5

    grouped = (
        df.groupby(["high_level_label", "decade"])
        .size()
        .reset_index(name="count")
        .pivot(index="decade", columns="high_level_label", values="count")
        .fillna(0)
    )

    grouped = grouped[grouped.sum(axis=1) >= 5]
    grouped = grouped.div(grouped.sum(axis=1), axis=0)

    grouped["solidarity/anti-solidarity"] = grouped["solidarity"] / grouped["anti-solidarity"]
    grouped["solidarity-anti-solidarity"] = grouped["solidarity"] - grouped["anti-solidarity"]

    for column in list(grouped.columns):
        grouped[f"smooth-{column}"] = gaussian_filter1d(
            grouped[column].interpolate(limit_area="inside"),
            sigma=1,
        )

    return grouped


def plot_data(dft: pd.DataFrame, output_path: Path) -> None:
    """Plot the decade-level difference between solidarity and anti-solidarity."""
    fig, ax = plt.subplots(figsize=(7.2 * 0.55, 3.0))
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.83)

    palette = sns.color_palette()

    ax.bar(
        dft.index,
        dft["solidarity-anti-solidarity"],
        width=7,
        alpha=0.4,
        linewidth=0,
        color=palette[7],
    )
    ax.plot(dft.index, dft["smooth-solidarity-anti-solidarity"], color="black")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Comparison of Solidarity and Anti-Solidarity")
    ax.set_xlabel("Year")
    ax.set_ylabel("Difference in Solidarity")
    ax.yaxis.set_label_coords(-0.15, 0.52)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()


def main() -> None:
    df = load_data(INPUT_PATH)
    dft = prepare_data(df)
    plot_data(dft, OUTPUT_PATH)


if __name__ == "__main__":
    main()