import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from . import config


def plot_training_curves(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(baseline_history, improved_history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    b_epochs = range(1, len(baseline_history["train_loss"]) + 1)
    i_epochs = range(1, len(improved_history["train_loss"]) + 1)

    axes[0, 0].plot(b_epochs, baseline_history["train_loss"], label="Train")
    axes[0, 0].plot(b_epochs, baseline_history["val_loss"], label="Val")
    axes[0, 0].set_title("Baseline - Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(i_epochs, improved_history["train_loss"], label="Train")
    axes[0, 1].plot(i_epochs, improved_history["val_loss"], label="Val")
    axes[0, 1].set_title("Improved - Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(b_epochs, baseline_history["train_acc"], label="Train")
    axes[1, 0].plot(b_epochs, baseline_history["val_acc"], label="Val")
    axes[1, 0].set_title("Baseline - Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(i_epochs, improved_history["train_acc"], label="Train")
    axes[1, 1].plot(i_epochs, improved_history["val_acc"], label="Val")
    axes[1, 1].set_title("Improved - Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.suptitle("Baseline vs Improved CNN Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
