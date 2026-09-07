"""Export a deployable artefact from the DementiaNet pipeline.

Run this INSIDE the dementianet-voicescreen repo, in the conda env, after the
feature CSVs exist. It fits the chosen SVM on the development split only (the
locked test set is never touched here) and writes one file the inference
service can load.

    python export_model.py --out caesura_model.joblib

What goes in the artefact, and why:
  pipeline      StandardScaler + SVC(rbf), fitted on dev only.
  columns       Exact feature order used at fit time. The service rebuilds its
                feature vector against this list, so a column added or renamed
                upstream fails loudly instead of silently shifting values.
  medians       Dev-set medians, used to fill any feature the service cannot
                compute for a given clip -- same policy as split_features().
  dev_scores    decision_function values over the dev set. The service reports
                a new clip's PERCENTILE within this distribution rather than a
                probability: SVC was deliberately fitted without probability=True
                in this project, and Platt-scaling it after the fact would invent
                a calibration nobody has validated.
  meta          Provenance, so a stale artefact is identifiable at a glance.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dementianet.models.train import load_feature_table, split_features
from dementianet.utils import load_config, set_seed


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="caesura_model.joblib")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])

    df = load_feature_table()

    # Development split only. The locked test set stays locked.
    if "split" in df.columns:
        dev = df[df["split"] == "dev"].copy()
        if dev.empty:
            raise SystemExit(
                "No rows with split == 'dev'. Check make_split output before exporting."
            )
    else:
        raise SystemExit("manifest has no 'split' column; run make_split first.")

    X, y, _groups = split_features(dev)
    print(f"[export] fitting on {len(X)} dev clips, {X.shape[1]} features")

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    pipe.fit(X, y)

    dev_scores = pipe.decision_function(X)

    joblib.dump(
        {
            "pipeline": pipe,
            "columns": list(X.columns),
            "medians": X.median(numeric_only=True).to_dict(),
            "dev_scores": np.asarray(dev_scores, dtype=float),
            "meta": {
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_rev": git_rev(),
                "n_dev_clips": int(len(X)),
                "n_features": int(X.shape[1]),
                "opensmile_set": cfg["features"]["opensmile_feature_set"],
                "min_pause_ms": cfg["features"]["pause"]["min_pause_ms"],
                "silence_threshold_db": cfg["features"]["pause"]["silence_threshold_db"],
                "target_sample_rate": cfg["audio"]["target_sample_rate"],
            },
        },
        args.out,
    )
    print(f"[export] wrote {args.out}")
    print("[export] copy it into server/ and deploy.")


if __name__ == "__main__":
    main()
