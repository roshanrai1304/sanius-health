import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

from . import config
from .models import get_models


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []

    for entry in models:
        name = entry["name"]
        model = entry["model"]
        print(f"\n--- {name} ---")

        cv_acc = cross_val_score(model, X_train, y_train,
                                 cv=config.CV_FOLDS, scoring="accuracy")
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            cv_auc = cross_val_score(model, X_train, y_train,
                                     cv=config.CV_FOLDS, scoring="roc_auc")
            cv_auc_mean = cv_auc.mean()
            print(f"CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
            print(f"CV ROC-AUC:  {cv_auc_mean:.4f} (+/- {cv_auc.std():.4f})")
        else:
            cv_auc_mean = 0.0
            print(f"CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
            print("CV ROC-AUC:  N/A (no predict_proba)")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if has_proba:
            y_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = np.zeros(len(y_test))
            test_auc = 0.0

        res = {
            "name": name,
            "model": model,
            "cv_accuracy": cv_acc.mean(),
            "cv_auc": cv_auc_mean,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1": f1_score(y_test, y_pred, average="binary"),
            "roc_auc": test_auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        results.append(res)

        print(f"Test Accuracy:  {res['accuracy']:.4f}")
        print(f"Test Precision: {res['precision']:.4f}")
        print(f"Test Recall:    {res['recall']:.4f}")
        print(f"Test F1:        {res['f1']:.4f}")
        print(f"Test ROC-AUC:   {res['roc_auc']:.4f}")

    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

    best = max(results, key=lambda x: x["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best['name']} ({best['roc_auc']:.4f})")

    return results
