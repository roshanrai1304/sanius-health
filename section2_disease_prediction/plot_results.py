import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

from . import config


def plot_roc_curves(results, y_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, r in enumerate(results):
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"], ax=axes[i])
        axes[i].set_title(r["name"])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results, save_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(results, feature_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        model_result = next((r for r in results if r["name"] == name), None)
        if model_result is None:
            ax.set_visible(False)
            continue
        importances = model_result["model"].feature_importances_
        indices = np.argsort(importances)[-15:]
        ax.barh(range(len(indices)),
                importances[indices], color="steelblue")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f"{name} - Top Features")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
