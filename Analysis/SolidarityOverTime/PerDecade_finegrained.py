from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


sns.set_theme(style="whitegrid")

plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

INPUT_PATH = Path("../../Data/ModelPredictedData/Migrant18kGPT4_predicted.json")
OUTPUT_PATH = Path("../../Analysis/SolidarityOverTime/PerDecade_finegrained_Migrant.pdf")

SOLIDARITY_LABELS = [
    "s.group-based",
    "s.exchange-based",
    "s.compassionate",
    "s.empathic",
]

ANTI_SOLIDARITY_LABELS = [
    "as.group-based",
    "as.exchange-based",
    "as.compassionate",
    "as.empathic",
]

LABEL_DETAILS = {
    "group-based": {"color": "#007BA7", "line_style": "-", "label": "Group-based"},
    "exchange-based": {"color": "#D31D0D", "line_style": "--", "label": "Exchange-based"},
    "compassionate": {"color": "#4CAF50", "line_style": "-.", "label": "Compassionate"},
    "empathic": {"color": "#F9AA00", "line_style": (0, [5, 5]), "label": "Empathic"},
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file and keep only required columns."""
    df = pd.read_json(path)
    return df[["id", "year", "extracted_label_GPT4"]].copy()


def prepare_grouped_data(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Filter labels, aggregate by decade, and normalize to proportions."""
    filtered = df[df["extracted_label_GPT4"].isin(labels)].copy()
    filtered["decade"] = (filtered["year"] // 10) * 10 + 5

    grouped = (
        filtered.groupby(["extracted_label_GPT4", "decade"])
        .size()
        .reset_index(name="count")
        .pivot(index="decade", columns="extracted_label_GPT4", values="count")
        .fillna(0)
    )

    grouped = grouped.div(grouped.sum(axis=1), axis=0)
    return grouped


def plot_panel(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    labels: list[str],
    title: str,
    show_legend: bool = False,
    show_ylabel: bool = True,
    line_width: int = 3,
) -> None:
    """Plot one panel of fine-grained label distributions."""
    for label in labels:
        if label not in grouped.columns:
            continue

        subtype = label.split(".")[1]
        config = LABEL_DETAILS[subtype]

        smoothed_values = gaussian_filter1d(
            grouped[label].interpolate(limit_area="inside"),
            sigma=1,
        )

        ax.plot(
            grouped.index,
            smoothed_values,
            color=config["color"],
            linestyle=config["line_style"],
            linewidth=line_width,
            label=config["label"],
        )

    ax.axvspan(1933, 1949, color="gray", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    if show_ylabel:
        ax.set_ylabel("Percentage of Label")
    else:
        ax.set_ylabel("")

    if show_legend:
        ax.legend(loc="upper left", fontsize="small")


def main() -> None:
    df = load_data(INPUT_PATH)

    solidarity_grouped = prepare_grouped_data(df, SOLIDARITY_LABELS)
    anti_solidarity_grouped = prepare_grouped_data(df, ANTI_SOLIDARITY_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.16, top=0.83, wspace=0.2)

    plot_panel(
        axes[0],
        solidarity_grouped,
        SOLIDARITY_LABELS,
        title="Solidarity",
        show_legend=True,
        show_ylabel=True,
    )

    plot_panel(
        axes[1],
        anti_solidarity_grouped,
        ANTI_SOLIDARITY_LABELS,
        title="Anti-Solidarity",
        show_legend=False,
        show_ylabel=False,
    )

    fig.suptitle("Solidarity vs. Anti-Solidarity Frames Distribution per Decade", x=0.542)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()