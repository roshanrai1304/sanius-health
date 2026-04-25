import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

from . import config

LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                          "diabetic_retinopathy.csv")


def load_dataset():
    try:
        if os.path.exists(LOCAL_PATH):
            print(f"Loading from local file: {LOCAL_PATH}")
            df = pd.read_csv(LOCAL_PATH)
        else:
            print("Downloading Diabetic Retinopathy dataset from UCI...")
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
            df = pd.concat([dataset.data.targets, dataset.data.features], axis=1)
            df = df.rename(columns={df.columns[0]: "target"})
            os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
            df.to_csv(LOCAL_PATH, index=False)
            print(f"Dataset saved locally to: {LOCAL_PATH}")

        if "Class" in df.columns:
            df = df.rename(columns={"Class": "target"})

        cols = list(df.columns)
        seen = {}
        for i, col in enumerate(cols):
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
        df.columns = cols

        df["target"] = df["target"].astype(int)

        print(f"Loaded Diabetic Retinopathy dataset: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing values:\n{missing[missing > 0]}")
        return df
    except Exception as e:
        print(f"Failed to load Diabetic Retinopathy data: {e}")
        print("Falling back to sklearn Breast Cancer dataset...")
        return _load_fallback()


def _load_fallback():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = 1 - data.target
    print(f"Loaded Breast Cancer dataset (fallback): {df.shape}")
    return df
