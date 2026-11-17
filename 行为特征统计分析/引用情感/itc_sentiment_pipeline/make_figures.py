from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config


def make_figures():
    src = config.OUTPUT_DIR / "05_statistical_results.xlsx"
    if not src.exists():
        raise FileNotFoundError("Run: python run_pipeline.py analyze")
    fig_dir = config.OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    dist = pd.read_excel(src, sheet_name="sentiment_distribution")
    palette = {"negative": "#D65F5F", "neutral": "#7A7A7A", "positive": "#4C9F70"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=dist, x="label", y="sentence_instance_pct", hue="label", palette=palette, legend=False, ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    ax.set(xlabel="Sentiment", ylabel="Citation sentence instances (%)", title="Sentiment distribution in ITC records")
    fig.tight_layout(); fig.savefig(fig_dir / "figure_sentiment_distribution.png", dpi=300); plt.close(fig)

    combo = pd.read_excel(src, sheet_name="label_combination").set_index("final_label1")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(combo, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set(xlabel="Sentence side 2", ylabel="Sentence side 1", title="Sentiment-label combinations within ITC pairs")
    fig.tight_layout(); fig.savefig(fig_dir / "figure_pair_combination_heatmap.png", dpi=300); plt.close(fig)

    section = pd.read_excel(src, sheet_name="section_distribution")
    g = sns.catplot(
        data=section, x="section", y="pct", hue="label", col="side", kind="bar",
        palette=palette, height=4.3, aspect=1.15, sharey=True
    )
    g.set_axis_labels("Section", "Within-section percentage")
    g.set_titles("Citation sentence side {col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    g.figure.tight_layout(); g.figure.savefig(fig_dir / "figure_section_sentiment.png", dpi=300); plt.close(g.figure)

    sens = pd.read_excel(src, sheet_name="confidence_sensitivity")
    long = sens.melt(
        id_vars=["confidence_cutoff", "pair_n", "consistency_rate"],
        value_vars=["negative_pct", "neutral_pct", "positive_pct"],
        var_name="label", value_name="pct"
    )
    long["label"] = long["label"].str.replace("_pct", "", regex=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=long, x="confidence_cutoff", y="pct", hue="label", marker="o", palette=palette, ax=ax)
    ax.set(xlabel="Minimum confidence cutoff", ylabel="Sentence instances (%)", title="Confidence-threshold sensitivity")
    fig.tight_layout(); fig.savefig(fig_dir / "figure_confidence_sensitivity.png", dpi=300); plt.close(fig)
    print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    make_figures()
