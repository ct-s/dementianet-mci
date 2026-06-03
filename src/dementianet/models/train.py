"""Train and evaluate baseline classifiers with speaker-independent CV.

CRITICAL: split by speaker (GroupKFold), never by clip. DementiaNet has
multiple clips per individual; a random split would leak a speaker across
train/test and inflate accuracy. This is the main methodological trap in the
project -- handle it correctly and document it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dementianet.utils import load_config, resolve, set_seed

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm_rbf": SVC(kernel="rbf", probability=True),
    "gradient_boosting": GradientBoostingClassifier(),
}


def load_feature_table() -> pd.DataFrame:
    """Merge feature CSVs with metadata (label, speaker_id, time bucket).

    Expects:
      data/processed/pause_features.csv
      data/processed/acoustic_egemaps.csv
      data/external/dementianet_metadata.csv  (clip_path, speaker_id, label, time_bucket)
    """
    cfg = load_config()
    proc = resolve(cfg["paths"]["data_processed"])
    meta = pd.read_csv(resolve(cfg["paths"]["metadata"]))
    pauses = pd.read_csv(proc / "pause_features.csv")
    acoustic = pd.read_csv(proc / "acoustic_egemaps.csv")
    df = meta.merge(pauses, on="clip_path").merge(acoustic, on="clip_path")
    return df


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    set_seed(cfg["seed"])
    group_col = cfg["model"]["group_column"]
    folds = cfg["model"]["cv_folds"]

    drop = {"clip_path", "label", group_col, "time_bucket", "valid"}
    X = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes("number")
    y = df["label"]
    groups = df[group_col]

    cv = GroupKFold(n_splits=folds)
    results = []
    for name, clf in CLASSIFIERS.items():
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="roc_auc")
        results.append({"model": name, "auc_mean": scores.mean(), "auc_std": scores.std()})
        print(f"[{name}] AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    cfg = load_config()
    out = resolve(cfg["paths"]["results_tables"]) / "baseline_auc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = evaluate(load_feature_table())
    res.to_csv(out, index=False)
    print(f"[train] wrote results -> {out}")
