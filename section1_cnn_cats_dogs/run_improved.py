import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from section1_cnn_cats_dogs import config
from section1_cnn_cats_dogs.dataset import create_dataloaders
from section1_cnn_cats_dogs.model_improved import ImprovedCNN
from section1_cnn_cats_dogs.train import train_model
from section1_cnn_cats_dogs.evaluate import evaluate_model
from section1_cnn_cats_dogs.plot_results import plot_training_curves, plot_confusion_matrix


def main():
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print("=" * 60)
    print("IMPROVED MODEL (With Regularization Techniques)")
    print("=" * 60)
    print("Techniques: Data Augmentation, Dropout, BatchNorm,")
    print("            Weight Decay, Early Stopping, LR Scheduling")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(augment=True)

    model = ImprovedCNN().to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True,
    )

    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.DEVICE, config.NUM_EPOCHS_IMPROVED,
        scheduler=scheduler, patience=config.PATIENCE,
    )

    plot_training_curves(
        history, "Improved CNN (With Regularization)",
        os.path.join(config.RESULTS_DIR, "improved_curves.png"),
    )

    metrics, preds, labels = evaluate_model(model, val_loader, config.DEVICE)
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    plot_confusion_matrix(
        metrics["confusion_matrix"], ["Cat", "Dog"],
        "Improved CNN - Confusion Matrix",
        os.path.join(config.RESULTS_DIR, "improved_confusion.png"),
    )

    with open(os.path.join(config.RESULTS_DIR, "improved_history.json"), "w") as f:
        json.dump(history, f)

    torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, "improved_model.pth"))
    print("Improved model training complete.")


if __name__ == "__main__":
    main()
