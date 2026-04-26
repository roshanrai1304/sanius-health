# Sanius Health - ML Project

Two machine learning tasks: (1) CNN binary classifier for cats vs dogs with overfitting demonstration and fix, and (2) Diabetic retinopathy prediction model using the UCI Diabetic Retinopathy Debrecen dataset.

---

## Prerequisites

- Python 3.9 or higher
- One of: macOS (Apple Silicon or Intel), Windows 10/11, or Linux (Ubuntu/Debian/Fedora)

---

## Python Setup

### macOS

```bash
# Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify
python3 --version
pip3 --version
```

### Windows

1. Download the Python installer from https://www.python.org/downloads/
2. Run the installer. **Check "Add Python to PATH"** during installation.
3. Open Command Prompt or PowerShell and verify:

```cmd
python --version
pip --version
```

Alternatively, install via winget:

```cmd
winget install Python.Python.3.11
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Verify
python3 --version
pip3 --version
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 python3-pip -y

# Verify
python3 --version
pip3 --version
```

---

## Create a Virtual Environment (Recommended)

### macOS / Linux

```bash
cd "sanius health"
python3 -m venv venv
source venv/bin/activate
```

### Windows (Command Prompt)

```cmd
cd "sanius health"
python -m venv venv
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
cd "sanius health"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

To deactivate later, run `deactivate` on any platform.

---

## Installation

### macOS / Linux

```bash
cd "sanius health"
pip3 install -r requirements.txt
```

### Windows

```cmd
cd "sanius health"
pip install -r requirements.txt
```

### Verify GPU Support

**macOS (Apple Silicon — MPS):**

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Windows / Linux (NVIDIA GPU — CUDA):**

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If no GPU is available, the code automatically falls back to CPU.

---

## Section 1: CNN Cats vs Dogs (Overfitting Demo)

Trains a CNN on the CIFAR-10 dataset (cat and dog classes) to classify cats vs dogs. Demonstrates overfitting with a large baseline model, then applies 6 regularization techniques to fix it.

### Regularization Techniques Used

| Technique | Details |
|---|---|
| Data Augmentation | RandomHorizontalFlip, RandomRotation, ColorJitter, RandomCrop |
| Dropout | Dropout2d(0.25) in conv blocks, Dropout(0.5) in FC layer |
| Batch Normalization | After every Conv2d and first Linear layer |
| Weight Decay (L2) | Adam optimizer with weight_decay=1e-4 |
| Early Stopping | Patience=7 epochs, restores best weights |
| LR Scheduling | ReduceLROnPlateau(patience=3, factor=0.5) |

### Running

**macOS / Linux:**

```bash
# Train baseline model (fails to learn — stuck at ~50% acc)
python3 section1_cnn_cats_dogs/run_baseline.py

# Train improved model (regularized: ~83% train acc, ~85% val acc)
python3 section1_cnn_cats_dogs/run_improved.py

# Run both and generate side-by-side comparison plot
python3 section1_cnn_cats_dogs/run_comparison.py
```

**Windows:**

```cmd
python section1_cnn_cats_dogs\run_baseline.py
python section1_cnn_cats_dogs\run_improved.py
python section1_cnn_cats_dogs\run_comparison.py
```

The CIFAR-10 dataset downloads automatically on first run (~170MB).

### Output

Results are saved to `section1_cnn_cats_dogs/results/`:
- `baseline_curves.png` — Training vs validation loss/accuracy (overfitting visible)
- `improved_curves.png` — Training vs validation loss/accuracy (gap reduced)
- `comparison.png` — Side-by-side comparison of both models
- `baseline_confusion.png` / `improved_confusion.png` — Confusion matrices
- `baseline_model.pth` / `improved_model.pth` — Saved model weights

### Expected Results

| Model | Train Acc | Val Acc | Gap |
|---|---|---|---|
| Baseline (~22M params) | ~50% | ~50% | ~0% (fails to learn) |
| Improved (~322K params) | ~83% | ~85% | ~-2% (healthy generalization) |

---

## Section 2: Diabetic Retinopathy Prediction

Builds and compares 11 ML models (6 individual + 5 ensemble) to predict diabetic retinopathy using the UCI Diabetic Retinopathy Debrecen dataset (1,151 samples, 19 features).

### Models

**Individual:** Logistic Regression, Random Forest, XGBoost, SVM, KNN, MLP

**Ensemble:** Gradient Boosting, AdaBoost, Bagging, Voting (Soft), Stacking

### Running

**macOS / Linux:**

```bash
python3 section2_disease_prediction/run_pipeline.py
```

**Windows:**

```cmd
python section2_disease_prediction\run_pipeline.py
```

The dataset downloads automatically from UCI ML Repository via `ucimlrepo` on first run and is saved locally for subsequent runs.

### Output

Results are saved to `section2_disease_prediction/results/`:
- `eda/` — Exploratory data analysis plots (histograms, correlation heatmap, box plots, class distribution)
- `roc_curves.png` — ROC curves for all models
- `confusion_matrices.png` — Confusion matrices for all models
- `model_comparison.png` — Bar chart comparing accuracy, precision, recall, F1, AUC
- `feature_importance.png` — Top features from Random Forest and XGBoost

### Expected Results

Best model by ROC-AUC: Logistic Regression (~0.83). Best model by accuracy/F1: MLP (~0.74/0.76).

---

## Project Structure

```
sanius health/
├── README.md
├── requirements.txt
├── section1_cnn_cats_dogs/
│   ├── config.py              # Hyperparameters, device selection (MPS/CUDA/CPU)
│   ├── dataset.py             # CIFAR-10 cat/dog loading, transforms
│   ├── model_baseline.py      # Overfitting CNN (~22M params)
│   ├── model_improved.py      # Regularized CNN (~322K params)
│   ├── train.py               # Training loop with early stopping
│   ├── evaluate.py            # Metrics computation
│   ├── plot_results.py        # Training curves, comparison plots
│   ├── run_baseline.py        # Entry point: baseline model
│   ├── run_improved.py        # Entry point: improved model
│   └── run_comparison.py      # Entry point: side-by-side comparison
└── section2_disease_prediction/
    ├── config.py              # Seeds, paths, model params
    ├── load_data.py           # Diabetic Retinopathy UCI download
    ├── eda.py                 # Exploratory data analysis
    ├── preprocessing.py       # Feature engineering, scaling
    ├── models.py              # All model definitions
    ├── train_evaluate.py      # Cross-validation, metrics
    ├── plot_results.py        # ROC curves, confusion matrices
    └── run_pipeline.py        # Single entry point for full pipeline
```

---

## Troubleshooting

**MPS not available (macOS):** Ensure you have PyTorch >= 2.0 and macOS 12.3+. The code falls back to CPU automatically.

**CUDA not available (Windows/Linux):** Install the CUDA-enabled version of PyTorch from https://pytorch.org/get-started/locally/. The code falls back to CPU automatically.

**Dataset download fails (Section 1):** Check your internet connection. The CIFAR-10 dataset is ~170MB.

**Dataset download fails (Section 2):** The pipeline automatically falls back to sklearn's built-in Breast Cancer dataset. Requires `ucimlrepo` package (`pip install ucimlrepo`).

**Import errors:** Make sure you run scripts from the project root directory (`sanius health/`), not from inside the section folders.

**PowerShell execution policy error (Windows):** If `Activate.ps1` fails, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Permission denied (Linux):** If pip install fails, use `pip3 install --user -r requirements.txt` or use a virtual environment.
