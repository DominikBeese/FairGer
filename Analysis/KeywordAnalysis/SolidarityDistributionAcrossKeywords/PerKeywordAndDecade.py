from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


sns.set_theme(style="whitegrid")

INPUT_PATH = Path("../../Data/ModelPredictedData/Migrant18kGPT4_predicted.json")
OUTPUT_PATH = Path("../../Analysis/KeywordAnalysis/SolidarityDistributionAcrossKeywords/PerKeywordAndDecade-Migrant.pdf")

FIGSIZE = (10.0, 4.0)
NCOLS = 8
NROWS = 4
X_LIMITS = (1867, 2022)
X_TICKS = [1900, 1950, 2000]

SOLIDARITY_LABELS = {
    "s.group-based",
    "s.empathic",
    "s.compassionate",
    "s.exchange-based",
    "s.none",
}

ANTISOLIDARITY_LABELS = {
    "as.group-based",
    "as.empathic",
    "as.compassionate",
    "as.exchange-based",
    "as.none",
}

SHORTEN_TITLE = {
    "Sowjetzonenflüchtlinge": "Sowjetzonen…",
    "Bürgerkriegsflüchtlinge": "Bürgerkriegs…",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file and keep only required columns."""
    df = pd.read_json(path)
    return df[["id", "year", "keyword", "extracted_label_GPT4"]].copy()


def map_label(label: str) -> int | None:
    """Map extracted labels to numeric plotting labels."""
    if pd.isna(label):
        return None
    if label in SOLIDARITY_LABELS:
        return 0
    if label in ANTISOLIDARITY_LABELS:
        return 1
    return 2


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Aggregate labels by keyword and decade and compute smoothed series."""
    order = df.groupby("keyword").size().sort_values(ascending=False).index

    dft = df.copy()
    dft["numeric_label"] = dft["extracted_label_GPT4"].apply(map_label)
    dft = dft[dft["numeric_label"].notna()].copy()
    dft["decade"] = (dft["year"] // 10) * 10 + 5

    dft = (
        dft.groupby(["keyword", "numeric_label", "decade"])
        .size()
        .reset_index(name="count")
        .pivot(index="decade", columns=["keyword", "numeric_label"], values="count")
        .fillna(0)
    )

    for keyword in order:
        if keyword not in dft:
            continue

        dft[keyword] = dft[keyword][dft[keyword].sum(axis=1) >= 5]
        dft[keyword] = dft[keyword].div(dft[keyword].sum(axis=1), axis=0)

        for column in dft[keyword].columns:
            dft[(keyword, f"smooth-{column}")] = gaussian_filter1d(
                dft[keyword][column].interpolate(limit_area="inside"),
                sigma=1,
                truncate=1,
            )

    return dft, order


def plot_data(
    dft: pd.DataFrame,
    order: pd.Index,
    output_path: Path,
    ncols: int,
    nrows: int,
    ymax: float,
    title_size: int,
    plot_antisolidarity: bool,
    shorten_title: dict[str, str] | None = None,
) -> None:
    """Plot solidarity and anti-solidarity distributions by keyword and decade."""
    if shorten_title is None:
        shorten_title = {}

    fig, axes = plt.subplots(figsize=FIGSIZE, ncols=ncols, nrows=nrows)
    axes = list(axes.flat)
    palette = sns.color_palette()

    for i, keyword in enumerate(order):
        ax = axes[i]
        show_xlabel = i >= len(order) - ncols
        show_ylabel = i % ncols == 0

        if keyword in dft and 0 in dft[keyword]:
            ax.bar(
                dft[keyword][0].index,
                dft[keyword][0].values,
                width=7,
                color=palette[2],
                alpha=0.4,
                linewidth=0,
            )
            ax.plot(dft[keyword]["smooth-0"], color=palette[2])

        if plot_antisolidarity and keyword in dft and 1 in dft[keyword]:
            bottom_vals = dft[keyword][0].fillna(0).values if 0 in dft[keyword] else 0

            ax.bar(
                dft[keyword][1].index,
                dft[keyword][1].values,
                bottom=bottom_vals,
                width=7,
                color=palette[3],
                alpha=0.4,
                linewidth=0,
            )

            if "smooth-0" in dft[keyword] and "smooth-1" in dft[keyword]:
                ax.plot(dft[keyword]["smooth-0"] + dft[keyword]["smooth-1"], color=palette[3])
            elif "smooth-1" in dft[keyword]:
                ax.plot(dft[keyword]["smooth-1"], color=palette[3])

        ax.set_xlim(X_LIMITS)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=10)
        ax.set_xticks(X_TICKS)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"${shorten_title.get(keyword, keyword)}$", fontsize=title_size, pad=2)

        if show_xlabel:
            ax.set_xlabel("Year")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if show_ylabel:
            ax.set_ylabel("Percentage", fontsize=11.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for i in range(len(order), ncols * nrows):
        fig.delaxes(axes[i])

    plt.subplots_adjust(left=0.07, right=0.99, top=0.885, bottom=0.12, wspace=0.08, hspace=0.30)
    plt.suptitle("Solidarity Distribution per Keyword and Decade", fontsize="medium")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def main() -> None:
    df = load_data(INPUT_PATH)
    dft, order = prepare_data(df)

    plot_data(
        dft=dft,
        order=order,
        output_path=OUTPUT_PATH,
        ncols=NCOLS,
        nrows=NROWS,
        ymax=1.0,
        title_size=9,
        plot_antisolidarity=True,
        shorten_title=SHORTEN_TITLE,
    )


if __name__ == "__main__":
    main()