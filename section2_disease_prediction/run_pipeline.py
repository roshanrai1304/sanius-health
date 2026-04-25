import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from section2_disease_prediction import config
from section2_disease_prediction.load_data import load_dataset
from section2_disease_prediction.eda import run_eda
from section2_disease_prediction.preprocessing import preprocess
from section2_disease_prediction.train_evaluate import train_and_evaluate
from section2_disease_prediction.plot_results import (
    plot_roc_curves, plot_confusion_matrices,
    plot_model_comparison, plot_feature_importance,
)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EDA_DIR, exist_ok=True)

    print("=" * 60)
    print("DIABETIC RETINOPATHY PREDICTION PIPELINE")
    print("UCI Diabetic Retinopathy Debrecen Dataset")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_dataset()

    print("\n[2/4] Exploratory Data Analysis...")
    run_eda(df)

    print("\n[3/4] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("\n[4/4] Training & Evaluation...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nGenerating plots...")
    plot_roc_curves(results, y_test,
                    os.path.join(config.RESULTS_DIR, "roc_curves.png"))
    plot_confusion_matrices(results,
                            os.path.join(config.RESULTS_DIR, "confusion_matrices.png"))
    plot_model_comparison(results,
                          os.path.join(config.RESULTS_DIR, "model_comparison.png"))
    plot_feature_importance(results, list(X_train.columns),
                            os.path.join(config.RESULTS_DIR, "feature_importance.png"))

    print("\n" + "=" * 60)
    print("Pipeline complete! Results saved to:", config.RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
