import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from section1_cnn_cats_dogs import config
from section1_cnn_cats_dogs.dataset import create_dataloaders
from section1_cnn_cats_dogs.model_baseline import BaselineCNN
from section1_cnn_cats_dogs.train import train_model
from section1_cnn_cats_dogs.evaluate import evaluate_model
from section1_cnn_cats_dogs.plot_results import plot_training_curves, plot_confusion_matrix


def main():
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print("=" * 60)
    print("BASELINE MODEL (No Regularization - Will Overfit)")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(augment=False)

    model = BaselineCNN().to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.DEVICE, config.NUM_EPOCHS_BASELINE,
    )

    plot_training_curves(
        history, "Baseline CNN (Overfitting)",
        os.path.join(config.RESULTS_DIR, "baseline_curves.png"),
    )

    metrics, preds, labels = evaluate_model(model, val_loader, config.DEVICE)
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    plot_confusion_matrix(
        metrics["confusion_matrix"], ["Cat", "Dog"],
        "Baseline CNN - Confusion Matrix",
        os.path.join(config.RESULTS_DIR, "baseline_confusion.png"),
    )

    with open(os.path.join(config.RESULTS_DIR, "baseline_history.json"), "w") as f:
        json.dump(history, f)

    torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, "baseline_model.pth"))
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
