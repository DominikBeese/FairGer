from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


sns.set_theme(style="whitegrid")

INPUT_PATH = Path("../../Data/Datasets/Migrant.json")
OUTPUT_PATH = Path("../../Analysis/KeywordAnalysis/KeywordDistributionOverTime/DistributionOfMigrantKeywordsPerYear.pdf")

FIGSIZE = (10.0, 4.0)
NROWS = 4
NCOLS = 8
X_LIMITS = (1867, 2022)
Y_LIMITS = (0, 0.13)
X_TICKS = [1900, 1950, 2000]
SMOOTHING_SIGMA = 3

SHORTENED_TITLES = {
    "Sowjetzonenflüchtlinge": "Sowjetzonen…",
    "Bürgerkriegsflüchtlinge": "Bürgerkriegs…",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file and keep only required columns."""
    df = pd.read_json(path)
    return df[["id", "year", "month", "day", "category", "keyword"]].copy()


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Aggregate keyword frequencies by year and compute smoothed series."""
    grouped = (
        df.groupby(["keyword", "year"])
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="keyword", values="count")
        .fillna(0)
    )

    grouped = grouped.reindex(range(grouped.index.min(), grouped.index.max() + 1)).fillna(0)

    order = grouped.sum(axis=0).sort_values(ascending=False).index
    grouped = grouped.div(grouped.sum(axis=0), axis=1)

    for keyword in grouped.columns:
        grouped[f"smooth-{keyword}"] = gaussian_filter1d(grouped[keyword], sigma=SMOOTHING_SIGMA)

    return grouped, order


def plot_data(dft: pd.DataFrame, order: pd.Index, output_path: Path) -> None:
    """Plot yearly keyword distributions in a grid."""
    fig, axes = plt.subplots(figsize=FIGSIZE, ncols=NCOLS, nrows=NROWS)
    axes = list(axes.flat)

    palette = sns.color_palette()

    for i, keyword in enumerate(order):
        ax = axes[i]
        show_xlabel = i >= NCOLS * (NROWS - 1)
        show_ylabel = i % NCOLS == 0

        ax.plot(dft.index, dft[f"smooth-{keyword}"], color=palette[0])
        ax.bar(
            dft.index,
            dft[keyword],
            width=0.7,
            color=palette[7],
            alpha=0.4,
            linewidth=0,
        )

        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.tick_params(labelsize=10)
        ax.set_xticks(X_TICKS)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        ax.set_title(f"${SHORTENED_TITLES.get(keyword, keyword)}$", fontsize=9, pad=2)

        if show_xlabel:
            ax.set_xlabel("Year")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if show_ylabel:
            ax.set_ylabel("Popularity")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for i in range(len(order), NROWS * NCOLS):
        fig.delaxes(axes[i])

    plt.subplots_adjust(left=0.07, right=0.99, top=0.885, bottom=0.12, wspace=0.08, hspace=0.30)
    plt.suptitle("Distribution of Keywords per Year", fontsize="medium")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def main() -> None:
    df = load_data(INPUT_PATH)
    dft, order = prepare_data(df)
    plot_data(dft, order, OUTPUT_PATH)


if __name__ == "__main__":
    main()