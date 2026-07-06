"""Feature-group ablation to attribute AUC to feature families.

Removing several features at once (as we did with duration + raw counts)
confounds the before/after comparison. This varies ONE family at a time,
on the DEV set with gradient boosting + speaker-grouped CV, so you can see
which features actually carry the signal.

Run:  python -m dementianet.models.ablation
Output: results/tables/ablation.csv
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dementianet.models.train import NON_FEATURE_COLS, load_feature_table
from dementianet.utils import load_config, resolve, set_seed

# Feature families (by name). Anything not listed here is treated as acoustic.
DURATION = ["duration_s"]
COUNTS = ["n_pauses", "n_long_pauses", "n_short_pauses"]
RATES = ["pause_rate", "long_pause_rate", "short_pause_rate"]
PAUSE_OTHER = ["psd", "mean_pause_s", "std_pause_s", "long_pause_ratio", "articulation_rate"]
FILLED = [
    "n_filled_pauses",
    "filled_pause_rate",
    "filled_to_silent_ratio",
    "filled_pause_time_ratio",
]


def all_feature_cols(df) -> list[str]:
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    return X.select_dtypes("number").drop(columns=["y"], errors="ignore").columns.tolist()


def cv_auc(df, cols) -> tuple[float, float, int]:
    cfg = load_config()
    set_seed(cfg["seed"])
    cols = [c for c in cols if c in df.columns]
    X = df[cols].fillna(df[cols].median(numeric_only=True))
    y = df["y"]
    groups = df[cfg["model"]["group_column"]]
    scores = cross_val_score(
        make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42)),
        X,
        y,
        groups=groups,
        cv=GroupKFold(cfg["model"]["cv_folds"]),
        scoring="roc_auc",
    )
    return scores.mean(), scores.std(), len(cols)


def main():
    cfg = load_config()
    df = load_feature_table()
    df = df[df["split"] == "dev"]

    known = set(DURATION + COUNTS + RATES + PAUSE_OTHER + FILLED)
    acoustic = [c for c in all_feature_cols(df) if c not in known]
    clean_pause = RATES + PAUSE_OTHER  # duration-free, non-redundant pause block

    configs = {
        "clean pause + acoustic (baseline)": clean_pause + acoustic,
        "acoustic only (no pause)": acoustic,
        "pause only (no acoustic)": clean_pause,
        # Filled-pause block: does it add signal over the baseline?
        "+ filled pauses (full)": clean_pause + acoustic + FILLED,
        "clean pause + filled (no acoustic)": clean_pause + FILLED,
        "filled pauses only": FILLED,
    }
    # Keep only configs whose features exist (filled pauses are optional).
    configs = {n: [c for c in cols if c in df.columns] for n, cols in configs.items()}
    configs = {n: cols for n, cols in configs.items() if cols}

    rows = []
    for name, cols in configs.items():
        m, s, k = cv_auc(df, cols)
        rows.append(
            {"config": name, "n_features": k, "auc_mean": round(m, 3), "auc_std": round(s, 3)}
        )
        print(f"{name:48s} k={k:3d}  AUC = {m:.3f} +/- {s:.3f}")

    out = resolve(cfg["paths"]["results_tables"]) / "ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[ablation] wrote -> {out}")


if __name__ == "__main__":
    main()
