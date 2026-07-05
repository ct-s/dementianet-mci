"""Assign a fixed, speaker-level dev/test split, written to the manifest.

Rules that keep the evaluation honest:
  - split at the SPEAKER level (never split one person across dev/test) -> no leakage
  - stratify by label so class balance is preserved in both sets
  - deterministic (split_seed) so the partition never drifts between runs

Run ONCE:  python -m dementianet.data.make_split
Rerun only if you deliberately want a new partition (this changes your test set).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from dementianet.utils import load_config, resolve


def make_split() -> pd.DataFrame:
    cfg = load_config()
    seed = cfg["dataset"].get("split_seed", cfg["seed"])
    test_size = cfg["dataset"].get("test_size", 0.25)
    manifest_path = resolve(cfg["dataset"]["manifest"])

    df = pd.read_csv(manifest_path)
    if "dx" not in df.columns:
        raise KeyError("Run enrich_manifest before make_split (need dx/label).")

    # One row per speaker, with their (single) label -> stratified speaker split.
    speakers = df[["speaker_id", "label"]].drop_duplicates("speaker_id")
    _, test_speakers = train_test_split(
        speakers["speaker_id"],
        test_size=test_size,
        random_state=seed,
        stratify=speakers["label"],
    )
    test_set = set(test_speakers)
    df["split"] = df["speaker_id"].apply(lambda s: "test" if s in test_set else "dev")

    # Leakage guard: no speaker may appear in both splits.
    overlap = set(df.loc[df.split == "dev", "speaker_id"]) & test_set
    assert not overlap, f"speaker leakage across split: {overlap}"

    df.to_csv(manifest_path, index=False)
    for name in ("dev", "test"):
        sub = df[df.split == name]
        print(
            f"[{name}] {sub['speaker_id'].nunique()} speakers | {len(sub)} clips "
            f"| labels: {sub['label'].value_counts().to_dict()}"
        )
    print(f"[ok] no speaker in both splits; written -> {manifest_path}")
    return df


if __name__ == "__main__":
    make_split()
