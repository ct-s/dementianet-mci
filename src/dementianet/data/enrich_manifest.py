"""Add a `dx` (diagnosis) column to the clip-level manifest.

Joins each speaker to their `dementia type` in the DementiaNet metadata
(matched on a normalized name key), and labels controls as "control".
Idempotent: recomputes `dx` from label/speaker each run, so it's safe to rerun.

Run:  python -m dementianet.data.enrich_manifest
"""

from __future__ import annotations

import csv
import re

import pandas as pd

from dementianet.utils import load_config, resolve


def norm(s: str) -> str:
    """Normalize a name for joining: lowercase, letters only."""
    return re.sub(r"[^a-z]", "", str(s).lower())


def build_dx_map(metadata_path) -> dict[str, str]:
    """name-key -> dementia type, from the (dementia-only) metadata CSV."""
    dx_map: dict[str, str] = {}
    with open(metadata_path) as f:
        for row in csv.DictReader(f):
            dx_map[norm(row["name"])] = (row["dementia type"] or "").strip()
    return dx_map


def enrich() -> pd.DataFrame:
    cfg = load_config()
    manifest_path = resolve(cfg["dataset"]["manifest"])
    dx_map = build_dx_map(resolve(cfg["paths"]["metadata"]))

    df = pd.read_csv(manifest_path)
    df["dx"] = df.apply(
        lambda r: "control"
        if r["label"] == "nodementia"
        else dx_map.get(norm(r["speaker_id"]), "unknown"),
        axis=1,
    )

    # Sanity check: every dementia speaker should have matched a metadata entry.
    unknown = df[df["dx"] == "unknown"]["speaker_id"].unique()
    if len(unknown):
        print(f"[WARN] {len(unknown)} unmatched dementia speakers: {list(unknown)}")
    else:
        print("[ok] all dementia speakers matched to a diagnosis")

    df.to_csv(manifest_path, index=False)
    print(f"[enrich] wrote dx for {len(df)} clips -> {manifest_path}")
    print("\n[dx distribution]")
    print(df["dx"].value_counts().to_string())
    return df


if __name__ == "__main__":
    enrich()
