from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


sns.set_theme(style="whitegrid")

plt.rcParams.update(
    {
        "axes.titlesize": 24,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)

INPUT_PATH = Path("../../Data/ModelPredictedData/Migrant18kGPT4_predicted.json")
OUTPUT_PATH = Path("../../Analysis/SolidarityOverTime/PerDecade_highlevel_Migrant.pdf")

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

PLOT_CONFIG = {
    "solidarity": {
        "color": "#4CAF50",
        "marker": "o",
        "label": "Solidarity",
    },
    "anti-solidarity": {
        "color": "#FF9800",
        "marker": "^",
        "label": "Anti-solidarity",
    },
    "mixed": {
        "color": "#9C27B0",
        "marker": "+",
        "label": "Mixed",
    },
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file and keep only the columns needed for plotting."""
    df = pd.read_json(path)
    return df[["id", "year", "extracted_label_GPT4"]].copy()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map detailed labels to high-level categories and compute decade shares."""
    df["high_level_label"] = df["extracted_label_GPT4"].map(LABEL_MAP)
    df = df[df["high_level_label"].notna()].copy()

    df["decade"] = (df["year"] // 10) * 10

    shares = (
        df.groupby(["high_level_label", "decade"])
        .size()
        .reset_index(name="count")
        .pivot(index="decade", columns="high_level_label", values="count")
        .fillna(0)
    )

    shares = shares.div(shares.sum(axis=1), axis=0)

    for column in list(shares.columns):
        shares[f"smooth_{column}"] = gaussian_filter1d(
            shares[column].interpolate(limit_area="inside"),
            sigma=1,
        )

    return shares


def plot_data(dft: pd.DataFrame, output_path: Path) -> None:
    """Plot smoothed decade-level category shares."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.82)

    for category, config in PLOT_CONFIG.items():
        smooth_col = f"smooth_{category}"
        if smooth_col not in dft.columns:
            continue

        ax.plot(
            dft.index,
            dft[smooth_col],
            linestyle="-",
            linewidth=5,
            color=config["color"],
            marker=config["marker"],
            markersize=10,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=config["color"],
            label=config["label"],
        )

    ax.axvspan(1933, 1949, color="gray", alpha=0.2)
    ax.axvline(x=1933, color="gray", linestyle="--", linewidth=1)
    ax.axvline(x=1949, color="gray", linestyle="--", linewidth=1)

    ax.set_xlim(1860, 2022)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    ax.set_title("Solidarity Distribution per Decade (Migrant)", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage")
    ax.legend(loc="upper left", fontsize="medium")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.show()


def main() -> None:
    df = load_data(INPUT_PATH)
    dft = prepare_data(df)
    plot_data(dft, OUTPUT_PATH)


if __name__ == "__main__":
    main()