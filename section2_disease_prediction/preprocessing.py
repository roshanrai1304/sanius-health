import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


def preprocess(df):
    df = df.copy()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    ma_cols = [c for c in df.columns if c.startswith("ma") and c != "macula_opticdisc_distance"]
    if ma_cols:
        df["ma_total"] = df[ma_cols].sum(axis=1)

    exudate_cols = [c for c in df.columns if c.startswith("exudate")]
    if exudate_cols:
        df["exudate_total"] = df[exudate_cols].sum(axis=1)

    if "macula_opticdisc_distance" in df.columns and "opticdisc_diameter" in df.columns:
        df["macula_disc_ratio"] = df["macula_opticdisc_distance"] / (df["opticdisc_diameter"] + 1e-8)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED, stratify=y,
    )

    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    return X_train, X_test, y_train, y_test, scaler
