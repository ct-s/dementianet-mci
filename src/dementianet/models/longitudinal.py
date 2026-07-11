"""Longitudinal analysis: does the AD speech signal weaken before diagnosis?

Clip filenames encode time-before-diagnosis as <name>_<bin>[_<clipnum>], with
bin in {0, 5, 10, 15} meaning post-diagnosis, 0-5, 5-10, 10-15 years before.

Method (leakage-safe, no data excluded):
  1. Fit the clean model (pause rates + pause-other + acoustic) on the DEV set
     with speaker-grouped CV, collecting OUT-OF-FOLD probabilities for every clip.
  2. For each dementia time bin, compute AUC of {that bin} vs {all controls}
     using those OOF probabilities.

If the signal is real and emerges near diagnosis, AUC should be highest for
post/0-5yr and decline toward 10-15yr (more overlap with controls early on).

Run:  python -m dementianet.models.longitudinal
Outputs: results/tables/longitudinal_auc.csv, results/figures/longitudinal_auc.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dementianet.models.ablation import FILLED, PAUSE_OTHER, RATES
from dementianet.models.train import load_feature_table, split_features
from dementianet.utils import load_config, resolve, set_seed

_BIN = re.compile(r"_(0|5|10|15)(?:_\d+)?$")
_LABEL = {0: "post", 5: "0-5yr", 10: "5-10yr", 15: "10-15yr"}
_ORDER = ["post", "0-5yr", "5-10yr", "10-15yr"]


def _oof_proba(X, y, groups, folds):
    """Speaker-grouped out-of-fold P(dementia) for a given feature matrix."""
    clf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))
    return cross_val_predict(
        clf, X, y, groups=groups, cv=GroupKFold(folds), method="predict_proba"
    )[:, 1]


def add_time_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with its time-before-diagnosis bin (controls -> 'control')."""

    def tag(row):
        if row["dx"] == "control":
            return "control"
        m = _BIN.search(Path(row["clip_path"]).stem)
        return _LABEL[int(m.group(1))] if m else "unknown"

    df = df.copy()
    df["time_bin"] = df.apply(tag, axis=1)
    return df


def family_gradient(dev, X, y, groups, folds, figs, tabs):
    """Exploratory: per-time-bin AUC for each feature family separately.

    Shows whether the signal source shifts over time (voice quality vs pause
    timing). NOTE: per-bin n is small (~17-26), so read gross patterns only.
    """
    pause_set = set(RATES + PAUSE_OTHER)
    families = {
        "acoustic": [c for c in X.columns if c not in set(RATES + PAUSE_OTHER + FILLED)],
        "pause": [c for c in X.columns if c in pause_set],
        "pause+acoustic": list(X.columns),
    }

    long = []
    for fam, cols in families.items():
        if not cols:
            continue
        d = dev.assign(_p=_oof_proba(X[cols], y, groups, folds))
        ctrl = d[d["dx"] == "control"]
        for b in _ORDER:
            sub = d[d["time_bin"] == b]
            if len(sub) < 3:
                continue
            yy = [1] * len(sub) + [0] * len(ctrl)
            pp = list(sub["_p"]) + list(ctrl["_p"])
            long.append(
                {
                    "family": fam,
                    "time_bin": b,
                    "n": len(sub),
                    "auc": round(roc_auc_score(yy, pp), 3),
                }
            )

    res = pd.DataFrame(long)
    res.to_csv(tabs / "longitudinal_by_family.csv", index=False)
    print("\n[per-family AUC by time bin (exploratory, small n)]")
    print(res.pivot(index="time_bin", columns="family", values="auc").reindex(_ORDER).to_string())

    plt.figure(figsize=(6, 4))
    for fam in families:
        g = res[res["family"] == fam]
        if len(g):
            plt.plot(g["time_bin"], g["auc"], "o-", label=fam)
    plt.axhline(0.5, ls="--", c="grey", lw=1)
    plt.ylim(0.4, 1.0)
    plt.ylabel("AUC vs controls")
    plt.xlabel("time before diagnosis")
    plt.title("Signal source over time (exploratory, small n)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / "longitudinal_by_family.png", dpi=150)
    plt.close()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    folds = cfg["model"]["cv_folds"]

    df = add_time_bin(load_feature_table())
    dev = df[df["split"] == "dev"].copy()
    X, y, groups = split_features(dev)

    # Out-of-fold probabilities (speaker-grouped) -> no leakage.
    dev["proba"] = _oof_proba(X, y, groups, folds)

    controls = dev[dev["dx"] == "control"]
    print(f"[data] dev: {len(dev)} clips | {len(controls)} controls")

    def auc_vs_controls(subset):
        yy = [1] * len(subset) + [0] * len(controls)
        pp = list(subset["proba"]) + list(controls["proba"])
        return roc_auc_score(yy, pp)

    rows = []
    for b in _ORDER:
        sub = dev[dev["time_bin"] == b]
        if len(sub) < 3:
            continue
        rows.append(
            {
                "time_bin": b,
                "n_dementia": len(sub),
                "auc_vs_control": round(auc_vs_controls(sub), 3),
            }
        )

    # Summary contrasts.
    near = dev[dev["time_bin"].isin(["post", "0-5yr"])]
    far = dev[dev["time_bin"].isin(["5-10yr", "10-15yr"])]
    alld = dev[(dev["dx"] != "control") & (dev["time_bin"] != "unknown")]
    for name, sub in [
        ("NEAR (post+0-5yr)", near),
        ("FAR (5-10+10-15yr)", far),
        ("ALL dementia", alld),
    ]:
        rows.append(
            {
                "time_bin": name,
                "n_dementia": len(sub),
                "auc_vs_control": round(auc_vs_controls(sub), 3),
            }
        )

    res = pd.DataFrame(rows)
    print("\n[longitudinal AUC: dementia bin vs all controls]")
    print(res.to_string(index=False))

    tabs = resolve(cfg["paths"]["results_tables"])
    tabs.mkdir(parents=True, exist_ok=True)
    figs = resolve(cfg["paths"]["results_figures"])
    figs.mkdir(parents=True, exist_ok=True)
    res.to_csv(tabs / "longitudinal_auc.csv", index=False)

    # Plot the per-bin gradient (ordered along the timeline).
    grad = res[res["time_bin"].isin(_ORDER)]
    plt.figure(figsize=(6, 4))
    plt.plot(grad["time_bin"], grad["auc_vs_control"], "o-")
    plt.axhline(0.5, ls="--", c="grey", lw=1)
    plt.ylim(0.4, 1.0)
    plt.ylabel("AUC vs controls")
    plt.xlabel("time before diagnosis")
    plt.title("Does the AD speech signal weaken before diagnosis?")
    plt.tight_layout()
    plt.savefig(figs / "longitudinal_auc.png", dpi=150)
    plt.close()
    print(f"\n[longitudinal] wrote {tabs/'longitudinal_auc.csv'} and {figs/'longitudinal_auc.png'}")

    # Exploratory: how each feature family's signal changes across the timeline.
    family_gradient(dev, X, y, groups, folds, figs, tabs)


if __name__ == "__main__":
    main()
