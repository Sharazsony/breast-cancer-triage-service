"""Export the scikit-learn breast cancer dataset to data/raw/breast_cancer.csv.

Feature columns use the dataset feature names with spaces replaced by
underscores. The target column keeps the scikit-learn encoding:
0 = malignant, 1 = benign.

Usage:
    python -m training.prepare_data                  # v1: all 569 rows
    python -m training.prepare_data --drop-last 30   # v2: "corrected export"
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

EXPECTED_FEATURES = 30


def build_frame() -> pd.DataFrame:
    data = load_breast_cancer()
    columns = [name.replace(" ", "_") for name in data.feature_names]
    df = pd.DataFrame(data.data, columns=columns)
    df["target"] = data.target
    return df


def validate(df: pd.DataFrame) -> None:
    n_features = df.shape[1] - 1
    if n_features != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, found {n_features}")
    nulls = int(df.isnull().sum().sum())
    if nulls:
        raise ValueError(f"Dataset contains {nulls} null values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/breast_cancer.csv")
    parser.add_argument("--drop-last", type=int, default=0,
                        help="Remove the last N rows (simulates a corrected export)")
    args = parser.parse_args()

    df = build_frame()
    if args.drop_last:
        df = df.iloc[: -args.drop_last]
    validate(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows, {df.shape[1] - 1} features + target, 0 nulls")
    print(df["target"].map({0: "malignant", 1: "benign"}).value_counts().to_string())


if __name__ == "__main__":
    main()
