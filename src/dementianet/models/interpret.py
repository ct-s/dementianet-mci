"""Feature informativeness and confound checks.

Answers two questions about the pause + acoustic baseline:
  1. WHICH features drive the classifier? (gradient-boosting importance +
     permutation importance on a speaker-held-out split; optional SHAP)
  2. Are we confounded by clip DURATION? (duration-only AUC, and correlation
     of each feature with duration)

Run:  python -m dementianet.models.interpret
Outputs: results/tables/feature_importance.csv, results/tables/confound_report.csv,
         results/figures/feature_importance.png, results/figures/duration_by_class.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

from dementianet.models.train import load_feature_table, split_features
from dementianet.utils import load_config, resolve, set_seed


def feature_importance(X, y, groups, top_n: int = 20) -> pd.DataFrame:
    """GB impurity importance + permutation importance on a speaker-held-out test split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    clf = GradientBoostingClassifier(random_state=42).fit(X.iloc[train_idx], y.iloc[train_idx])

    perm = permutation_importance(
        clf,
        X.iloc[test_idx],
        y.iloc[test_idx],
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
    )
    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "gb_impurity": clf.feature_importances_,
            "perm_importance": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_importance", ascending=False)
    return imp.head(top_n).reset_index(drop=True)


def try_shap(X, y, groups, out_dir):
    """Optional SHAP summary; silently skipped if shap isn't installed."""
    try:
        import shap
    except ImportError:
        print("[shap] not installed -- skipping (pip install shap to enable)")
        return
    clf = GradientBoostingClassifier(random_state=42).fit(X, y)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X)
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close()
    print(f"[shap] wrote {out_dir/'shap_summary.png'}")


def confound_report(df, X, y, groups) -> pd.DataFrame:
    """How much can DURATION alone predict, and how does each feature correlate with it?"""
    cfg = load_config()
    folds = cfg["model"]["cv_folds"]

    rows = []
    if "duration_s" in df.columns:
        dur = df[["duration_s"]].fillna(df["duration_s"].median())
        auc = cross_val_score(
            GradientBoostingClassifier(random_state=42),
            dur,
            y,
            groups=groups,
            cv=GroupKFold(folds),
            scoring="roc_auc",
        )
        print(
            f"[confound] duration-only AUC = {auc.mean():.3f} +/- {auc.std():.3f} "
            f"(near 0.5 => duration is not the driver)"
        )
        d = df["duration_s"]
        for f in X.columns:
            rows.append(
                {
                    "feature": f,
                    "corr_with_duration": np.corrcoef(X[f], d)[0, 1],
                    "corr_with_label": np.corrcoef(X[f], y)[0, 1],
                }
            )
    return pd.DataFrame(rows).sort_values("corr_with_duration", key=abs, ascending=False)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    figs = resolve(cfg["paths"]["results_figures"])
    figs.mkdir(parents=True, exist_ok=True)
    tabs = resolve(cfg["paths"]["results_tables"])
    tabs.mkdir(parents=True, exist_ok=True)

    df = load_feature_table()
    df = df[df["split"] == "dev"]  # never inspect the locked test set
    X, y, groups = split_features(df)
    print(f"[data] DEV only: {len(df)} clips | {X.shape[1]} features | {groups.nunique()} speakers")

    # 1. Feature importance
    imp = feature_importance(X, y, groups)
    imp.to_csv(tabs / "feature_importance.csv", index=False)
    print("\n[top features by permutation importance]")
    print(imp[["feature", "perm_importance", "gb_impurity"]].head(15).to_string(index=False))

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top["feature"], top["perm_importance"], xerr=top["perm_std"])
    plt.xlabel("Permutation importance (AUC drop)")
    plt.title("Top features (speaker-held-out)")
    plt.tight_layout()
    plt.savefig(figs / "feature_importance.png", dpi=150)
    plt.close()

    try_shap(X, y, groups, figs)

    # 2. Confounds
    conf = confound_report(df, X, y, groups)
    conf.to_csv(tabs / "confound_report.csv", index=False)

    if "duration_s" in df.columns:
        plt.figure(figsize=(5, 4))
        lab = y.map({1: "dementia", 0: "control"})
        data = [df["duration_s"][lab == c] for c in ["control", "dementia"]]
        plt.boxplot(data, labels=["control", "dementia"])
        plt.ylabel("clip duration (s)")
        plt.title("Duration by class (confound check)")
        plt.tight_layout()
        plt.savefig(figs / "duration_by_class.png", dpi=150)
        plt.close()

    print(f"\n[interpret] wrote tables -> {tabs} and figures -> {figs}")


if __name__ == "__main__":
    main()
