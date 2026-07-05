"""Train and evaluate baseline classifiers with speaker-independent CV.

CRITICAL: split by speaker (GroupKFold), never by clip. DementiaNet has
multiple clips per individual; a random split would leak a speaker across
train/test and inflate accuracy. This is the main methodological trap in the
project -- handle it correctly and document it.

Data flow:
  data/processed/manifest.csv        (clip_path, speaker_id, label, dx)
  data/processed/pause_features.csv  (clip_path, pause features)
  data/processed/acoustic_egemaps.csv (optional; enable via model.use_acoustic)

The dementia group is filtered to the AD-enriched phenotype (Alzheimer +
unspecified Dementia) via dataset.include_dx in config.yaml. The positive class
is dataset.positive_dx; controls are the negative class.
"""

from __future__ import annotations

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
    "svm_rbf": SVC(kernel="rbf"),  # roc_auc uses decision_function; no probability needed
    "gradient_boosting": GradientBoostingClassifier(),
}

# Columns that are identifiers/labels, never model inputs.
NON_FEATURE_COLS = {"clip_path", "speaker_id", "label", "dx", "time_bucket", "valid", "split"}


def load_feature_table() -> pd.DataFrame:
    """Load manifest + features, apply the AD inclusion filter, add a binary target.

    Returns a dataframe with feature columns plus: speaker_id, dx, and `y`
    (1 = dementia/AD-enriched, 0 = control).
    """
    cfg = load_config()
    proc = resolve(cfg["paths"]["data_processed"])

    manifest = pd.read_csv(resolve(cfg["dataset"]["manifest"]))
    if "dx" not in manifest.columns:
        raise KeyError(
            "manifest.csv has no 'dx' column. Enrich it with diagnosis first "
            "(see the join snippet in the project notes)."
        )

    pauses = pd.read_csv(proc / "pause_features.csv")
    df = manifest.merge(pauses, on="clip_path", how="inner")

    if cfg["model"].get("use_acoustic", False):
        acoustic = pd.read_csv(proc / "acoustic_egemaps.csv")
        df = df.merge(acoustic, on="clip_path", how="inner")

    # Inclusion filter (decided a priori) + binary target.
    include = set(cfg["dataset"]["include_dx"])
    positive = set(cfg["dataset"]["positive_dx"])
    df = df[df["dx"].isin(include)].copy()
    df["y"] = df["dx"].isin(positive).astype(int)

    # Drop clips whose features failed (e.g. silent/too-short).
    if "valid" in df.columns:
        df = df[df["valid"] != False]  # noqa: E712  (keep NaN/True, drop explicit False)

    if "split" not in df.columns:
        raise KeyError("manifest.csv has no 'split' column. Run make_split first.")

    return df


def split_features(df: pd.DataFrame):
    """Return (X, y, groups): numeric feature matrix, target, and speaker groups.

    Shared by evaluate() and the interpretability module so the feature set is
    defined in exactly one place.
    """
    cfg = load_config()
    group_col = cfg["model"]["group_column"]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    X = X.select_dtypes("number").drop(columns=["y"], errors="ignore")
    X = X.fillna(X.median(numeric_only=True))
    return X, df["y"], df[group_col]


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    set_seed(cfg["seed"])
    folds = cfg["model"]["cv_folds"]

    X, y, groups = split_features(df)

    n_pos, n_neg, n_spk = int(y.sum()), int((1 - y).sum()), groups.nunique()
    print(
        f"[data] {len(df)} clips | {n_pos} dementia / {n_neg} control "
        f"| {n_spk} speakers | {X.shape[1]} features"
    )

    cv = GroupKFold(n_splits=folds)
    results = []
    for name, clf in CLASSIFIERS.items():
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="roc_auc")
        results.append({"model": name, "auc_mean": scores.mean(), "auc_std": scores.std()})
        print(f"[{name}] AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
    return pd.DataFrame(results)


def evaluate_test(df: pd.DataFrame) -> pd.DataFrame:
    """Fit on the FULL dev set, evaluate ONCE on the locked test set.

    Call this only at the very end of the project. Every peek at the test set
    erodes its value as an unbiased estimate, so do not use it for tuning.
    """
    from sklearn.metrics import roc_auc_score

    cfg = load_config()
    set_seed(cfg["seed"])
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]
    X_tr, y_tr, _ = split_features(dev)
    X_te, y_te, _ = split_features(test)
    X_te = X_te[X_tr.columns]  # align columns

    print(
        f"[TEST] train on {len(dev)} dev clips, evaluate {len(test)} test clips "
        f"({y_te.sum()} dementia / {(1 - y_te).sum()} control)"
    )
    rows = []
    for name, clf in CLASSIFIERS.items():
        pipe = make_pipeline(StandardScaler(), clf).fit(X_tr, y_tr)
        scores = (
            pipe.predict_proba(X_te)[:, 1]
            if hasattr(pipe, "predict_proba")
            else pipe.decision_function(X_te)
        )
        auc = roc_auc_score(y_te, scores)
        rows.append({"model": name, "test_auc": auc})
        print(f"[TEST {name}] AUC = {auc:.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    cfg = load_config()
    tabs = resolve(cfg["paths"]["results_tables"])
    tabs.mkdir(parents=True, exist_ok=True)

    df = load_feature_table()
    dev = df[df["split"] == "dev"]

    print("=== Cross-validation on DEV set ===")
    res = evaluate(dev)
    res.to_csv(tabs / "baseline_auc.csv", index=False)
    print(f"[train] wrote dev CV results -> {tabs / 'baseline_auc.csv'}")

    if "--test" in sys.argv:
        print("\n=== LOCKED TEST evaluation (use once, at the end) ===")
        test_res = evaluate_test(df)
        test_res.to_csv(tabs / "test_auc.csv", index=False)
        print(f"[train] wrote test results -> {tabs / 'test_auc.csv'}")
    else:
        print("\n[note] test set NOT touched. Add --test only for your final run.")
