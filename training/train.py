"""Train and select the breast-cancer triage model.

Malignant is the positive class. The CSV keeps the scikit-learn encoding
(0 = malignant, 1 = benign), so the target is re-encoded here as
y = 1 for malignant and y = 0 for benign. predict_proba[:, 1] is therefore
the malignant probability that the API returns.

Usage:
    python -m training.train --data-path data/raw/breast_cancer.csv \
        --output-dir artifacts --model-version 1.0.0
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCORING = {"recall_malignant": "recall", "accuracy": "accuracy", "roc_auc": "roc_auc"}


def build_candidates(params: dict, seed: int) -> dict[str, Pipeline]:
    lr, rf = params["models"]["logistic_regression"], params["models"]["random_forest"]
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **lr)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=1, **rf)),
        ]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = yaml.safe_load(Path(args.params).read_text())
    seed = params["seed"]
    model_version = args.model_version or params["model_version"]

    df = pd.read_csv(args.data_path)
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].to_numpy(dtype=float)
    y = (df["target"] == 0).astype(int).to_numpy()  # 1 = malignant (positive class)
    print(f"Loaded {len(df)} rows, {len(feature_names)} features, {y.sum()} malignant")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)
    candidates = build_candidates(params, seed)
    results = {}
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        cv_metrics = {k: round(float(np.mean(scores[f"test_{k}"])), 4) for k in SCORING}

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_metrics = {
            "recall_malignant": round(float(recall_score(y_test, pred)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }
        results[name] = {"cv": cv_metrics, "test": test_metrics}

    best = max(results, key=lambda n: results[n]["cv"]["recall_malignant"])
    print(f"\n{'model':<22}{'cv_recall':>10}{'cv_acc':>9}{'cv_auc':>9}"
          f"{'test_recall':>13}{'test_acc':>10}{'test_auc':>10}")
    for name, r in results.items():
        mark = "  <== selected" if name == best else ""
        print(f"{name:<22}{r['cv']['recall_malignant']:>10.4f}{r['cv']['accuracy']:>9.4f}"
              f"{r['cv']['roc_auc']:>9.4f}{r['test']['recall_malignant']:>13.4f}"
              f"{r['test']['accuracy']:>10.4f}{r['test']['roc_auc']:>10.4f}{mark}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[best], out / "model.joblib")
    card = {
        "model_version": model_version,
        "model_family": best,
        "positive_class": "malignant",
        "feature_names": feature_names,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_metrics": results[best]["cv"],
        "test_metrics": results[best]["test"],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"selected": best, "test": results[best]["test"], "candidates": results}, indent=2))
    print(f"\nSaved {out / 'model.joblib'}, model_card.json and metrics.json "
          f"(version {model_version}, sklearn {sklearn.__version__})")

    recall = results[best]["test"]["recall_malignant"]
    if recall < params["min_test_recall"]:
        print(f"FAIL: test malignant recall {recall} < {params['min_test_recall']}", file=sys.stderr)
        return 1
    print(f"PASS: test malignant recall {recall} >= {params['min_test_recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
