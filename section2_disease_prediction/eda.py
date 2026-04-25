import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config


def run_eda(df):
    os.makedirs(config.EDA_DIR, exist_ok=True)

    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print(f"\nBasic statistics:\n{df.describe()}")

    numeric_cols = df.select_dtypes(include="number").columns.drop("target")
    n_cols = len(numeric_cols)
    cols_per_row = 5
    n_rows = math.ceil(n_cols / cols_per_row)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        for t in [0, 1]:
            subset = df[df["target"] == t][col].dropna()
            axes[i].hist(subset, alpha=0.5,
                         label=f"{'No DR' if t == 0 else 'DR'}", bins=20)
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions by Target", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.EDA_DIR, "histograms.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(config.EDA_DIR, "correlation.png"), dpi=150)
    plt.close()

    key_features = ["ma1", "ma2", "exudate1", "exudate2",
                     "macula_opticdisc_distance", "opticdisc_diameter"]
    available = [c for c in key_features if c in df.columns]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, col in zip(axes, available):
            sns.boxplot(data=df, x="target", y=col, ax=ax)
            ax.set_xticklabels(["No DR", "DR"])
            ax.set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(config.EDA_DIR, "boxplots.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    df["target"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_xticklabels(["No Retinopathy", "Retinopathy"], rotation=0)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(config.EDA_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    print(f"EDA plots saved to {config.EDA_DIR}")
