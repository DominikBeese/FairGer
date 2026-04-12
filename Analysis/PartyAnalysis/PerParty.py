from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

INPUT_PATH = Path("../../Data/ModelPredictedData/Migrant18kGPT4_predicted.json")
OUTPUT_PATH = Path("../../Analysis/PartyAnalysis/PerParty.pdf")

PARTY_ORDER = ["Linke", "Grüne", "SPD", "FDP", "CDU/CSU", "AfD"]

LABEL_MAP = {
    "s.group-based": "Solidarity",
    "s.exchange-based": "Solidarity",
    "s.compassionate": "Solidarity",
    "s.empathic": "Solidarity",
    "as.group-based": "Anti-Solidarity",
    "as.exchange-based": "Anti-Solidarity",
    "as.compassionate": "Anti-Solidarity",
    "as.empathic": "Anti-Solidarity",
}

SOLIDARITY_SUBTYPES = [
    "s.group-based",
    "s.exchange-based",
    "s.compassionate",
    "s.empathic",
]

ANTI_SOLIDARITY_SUBTYPES = [
    "as.group-based",
    "as.exchange-based",
    "as.compassionate",
    "as.empathic",
]

SUBTYPE_COLOR_MAP = {
    "s.group-based": "#007BA7",
    "s.exchange-based": "#D31D0D",
    "s.compassionate": "#4CAF50",
    "s.empathic": "#F9AA00",
    "as.group-based": "#007BA7",
    "as.exchange-based": "#D31D0D",
    "as.compassionate": "#4CAF50",
    "as.empathic": "#F9AA00",
}

LEGEND_LABELS = ["Group-based", "Exchange-based", "Compassionate", "Empathic"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the JSON file."""
    return pd.read_json(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map labels, filter parties, and compute subtype shares per party."""
    df = df.copy()
    df["Category"] = df["extracted_label_GPT4"].map(LABEL_MAP)
    df["Subtype"] = df["extracted_label_GPT4"]

    df = df[df["party"].isin(PARTY_ORDER)].copy()
    df["party"] = pd.Categorical(df["party"], categories=PARTY_ORDER, ordered=True)

    # Keep original behavior:
    # denominator = all statements per party, including rows not mapped in LABEL_MAP
    total_statements_per_party = df.groupby("party", observed=False).size()

    grouped = (
        df.groupby(["party", "Category", "Subtype"], observed=False)
        .size()
        .reset_index(name="Count")
    )

    grouped["Total Statements"] = grouped["party"].map(total_statements_per_party)
    grouped["Percent"] = grouped["Count"] / grouped["Total Statements"].astype(float)

    return grouped


def format_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply shared axis formatting."""
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    subtypes: list[str],
    title: str,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    """Plot one panel for a category."""
    sns.barplot(
        ax=ax,
        data=data,
        x="party",
        y="Percent",
        hue="Subtype",
        hue_order=subtypes,
        palette=SUBTYPE_COLOR_MAP,
    )

    format_axis(ax, title=title, ylabel="Percentage of Total" if show_ylabel else "")

    if show_legend:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, LEGEND_LABELS, title="Subtype", bbox_to_anchor=(1, 1))
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def main() -> None:
    df = load_data(INPUT_PATH)
    grouped = prepare_data(df)

    solidarity_df = grouped[grouped["Category"] == "Solidarity"]
    anti_solidarity_df = grouped[grouped["Category"] == "Anti-Solidarity"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    plot_panel(
        axes[0],
        solidarity_df,
        SOLIDARITY_SUBTYPES,
        title="Solidarity Subtypes Distribution",
        show_ylabel=True,
        show_legend=False,
    )

    plot_panel(
        axes[1],
        anti_solidarity_df,
        ANTI_SOLIDARITY_SUBTYPES,
        title="Anti-Solidarity Subtypes Distribution",
        show_ylabel=False,
        show_legend=True,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()