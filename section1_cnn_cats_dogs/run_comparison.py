import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from section1_cnn_cats_dogs import config
from section1_cnn_cats_dogs.dataset import create_dataloaders
from section1_cnn_cats_dogs.model_baseline import BaselineCNN
from section1_cnn_cats_dogs.model_improved import ImprovedCNN
from section1_cnn_cats_dogs.train import train_model
from section1_cnn_cats_dogs.evaluate import evaluate_model
from section1_cnn_cats_dogs.plot_results import plot_comparison, plot_confusion_matrix


def load_or_train_baseline(train_loader, val_loader):
    history_path = os.path.join(config.RESULTS_DIR, "baseline_history.json")
    if os.path.exists(history_path):
        print("Loading saved baseline history...")
        with open(history_path) as f:
            return json.load(f)

    print("Training baseline model...")
    model = BaselineCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.DEVICE, config.NUM_EPOCHS_BASELINE,
    )
    with open(history_path, "w") as f:
        json.dump(history, f)
    return history


def load_or_train_improved(train_loader, val_loader):
    history_path = os.path.join(config.RESULTS_DIR, "improved_history.json")
    if os.path.exists(history_path):
        print("Loading saved improved history...")
        with open(history_path) as f:
            return json.load(f)

    print("Training improved model...")
    model = ImprovedCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.DEVICE, config.NUM_EPOCHS_IMPROVED,
        scheduler=scheduler, patience=config.PATIENCE,
    )
    with open(history_path, "w") as f:
        json.dump(history, f)
    return history


def main():
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print("=" * 60)
    print("BASELINE vs IMPROVED CNN COMPARISON")
    print("=" * 60)

    train_loader_base, val_loader = create_dataloaders(augment=False)
    train_loader_aug, _ = create_dataloaders(augment=True)

    baseline_history = load_or_train_baseline(train_loader_base, val_loader)
    improved_history = load_or_train_improved(train_loader_aug, val_loader)

    plot_comparison(
        baseline_history, improved_history,
        os.path.join(config.RESULTS_DIR, "comparison.png"),
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    b_final_train = baseline_history["train_acc"][-1]
    b_final_val = baseline_history["val_acc"][-1]
    i_final_train = improved_history["train_acc"][-1]
    i_final_val = improved_history["val_acc"][-1]

    print(f"{'Model':<15} {'Train Acc':>10} {'Val Acc':>10} {'Gap':>10}")
    print("-" * 45)
    print(f"{'Baseline':<15} {b_final_train:>10.4f} {b_final_val:>10.4f} {b_final_train - b_final_val:>10.4f}")
    print(f"{'Improved':<15} {i_final_train:>10.4f} {i_final_val:>10.4f} {i_final_train - i_final_val:>10.4f}")
    print(f"\nOverfitting reduced by: {(b_final_train - b_final_val) - (i_final_train - i_final_val):.4f}")


if __name__ == "__main__":
    main()
