"""Export real model results for the static page.

Run this in the dementianet-voicescreen repo after export_model.py. It scores
the LOCKED TEST SET with the exported model and writes docs/cases.json, which
the page loads to show what the model actually does on real people -- including
the ones it gets wrong.

    python export_demo_cases.py --model caesura_model.joblib --out docs/cases.json

Why the test set and not the dev set: the test AUC is already published in the
README, so reporting the per-clip predictions behind that number reveals
nothing new and adds nothing to the multiple-comparisons budget. It is simply
showing the working behind a figure already claimed.

No audio is exported. Speakers are reduced to an opaque index, so nothing in
cases.json identifies the individuals in the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from dementianet.models.train import load_feature_table, split_features
from dementianet.utils import load_config

PAUSE_KEYS = [
    "psd",
    "pause_rate",
    "long_pause_rate",
    "short_pause_rate",
    "mean_pause_s",
    "std_pause_s",
    "long_pause_ratio",
    "articulation_rate",
]


def anon(speaker_id: str) -> str:
    """Stable, non-reversible label. Same speaker -> same tag across runs."""
    return "S" + hashlib.sha256(str(speaker_id).encode()).hexdigest()[:4].upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="caesura_model.joblib")
    ap.add_argument("--out", default="docs/cases.json")
    ap.add_argument("--n", type=int, default=12, help="how many cases to publish")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe, columns = bundle["pipeline"], bundle["columns"]
    dev_scores = np.sort(np.asarray(bundle["dev_scores"], dtype=float))
    cut = 0.67  # keep in step with CAESURA_TIER_CUT and the reviewer-panel copy

    df = load_feature_table()
    if "split" not in df.columns:
        raise SystemExit("manifest has no 'split' column; run make_split first.")
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("No rows with split == 'test'.")

    X, y, groups = split_features(test)
    X = X.reindex(columns=columns)
    X = X.fillna(X.median(numeric_only=True))

    decisions = pipe.decision_function(X)
    pct = np.searchsorted(dev_scores, decisions) / len(dev_scores)
    tier = np.where(pct >= cut, "elevated", "typical")
    truth = np.where(y.values == 1, "dementia", "control")
    # "Correct" here means the product's recommendation matched the group.
    correct = (tier == "elevated") == (y.values == 1)

    rows = []
    for i in range(len(X)):
        row = {
            "id": anon(groups.iloc[i]),
            "group": truth[i],
            "score": round(float(pct[i]), 3),
            "tier": str(tier[i]),
            "correct": bool(correct[i]),
            "pause": {
                k: round(float(test.iloc[i][k]), 4)
                for k in PAUSE_KEYS
                if k in test.columns and np.isfinite(test.iloc[i][k])
            },
        }
        rows.append(row)

    # Publish a balanced, honest sample: hits and misses from both groups, so
    # the page cannot accidentally flatter the model by cherry-picking.
    buckets = {
        ("control", True): [],
        ("control", False): [],
        ("dementia", True): [],
        ("dementia", False): [],
    }
    for r in rows:
        buckets[(r["group"], r["correct"])].append(r)
    for b in buckets.values():
        b.sort(key=lambda r: r["score"])

    per = max(1, args.n // 4)
    picked, seen = [], set()
    for key in [("dementia", True), ("control", True), ("dementia", False), ("control", False)]:
        for r in buckets[key][:per]:
            if r["id"] not in seen:
                picked.append(r)
                seen.add(r["id"])
    picked.sort(key=lambda r: r["score"])

    cfg = load_config()
    payload = {
        "cut": cut,
        "model": {
            "git_rev": bundle["meta"].get("git_rev"),
            "exported_at": bundle["meta"].get("exported_at"),
            "n_features": bundle["meta"].get("n_features"),
        },
        "cohort": {
            "n_test_clips": int(len(X)),
            "n_test_speakers": int(groups.nunique()),
            "n_correct": int(correct.sum()),
            "accuracy": round(float(correct.mean()), 3),
            "positive_dx": cfg["dataset"]["positive_dx"],
        },
        "cases": picked,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"[cases] {len(picked)} of {len(rows)} test clips -> {out}")
    print(
        f"[cases] recommendation matched the group in "
        f"{payload['cohort']['n_correct']}/{len(rows)} test clips "
        f"({payload['cohort']['accuracy']:.0%})"
    )
    print("[cases] no audio and no speaker names were written.")


if __name__ == "__main__":
    main()
