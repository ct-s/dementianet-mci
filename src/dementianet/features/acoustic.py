"""Acoustic feature extraction with openSMILE (eGeMAPS).

eGeMAPS gives ~88 functionals including jitter, shimmer, HNR, F0 statistics,
loudness, and spectral measures -- the voice-quality markers linked to AD in
the report (locus coeruleus / arousal axis: jitter & shimmer).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dementianet.utils import load_config, resolve


def extract_egemaps(interim_dir: Path | None = None) -> pd.DataFrame:
    """Run openSMILE eGeMAPS over every standardized wav; return one row per clip."""
    import opensmile

    cfg = load_config()
    interim = interim_dir or resolve(cfg["paths"]["data_interim"])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows = []
    for wav in sorted(Path(interim).rglob("*.wav")):
        feats = smile.process_file(str(wav))
        feats.insert(0, "clip_path", wav.name)  # basename, to match manifest & pause_features
        rows.append(feats.reset_index(drop=True))

    if not rows:
        raise FileNotFoundError(f"No wavs found in {interim}. Run download.py first.")
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    cfg = load_config()
    out = resolve(cfg["paths"]["data_processed"]) / "acoustic_egemaps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = extract_egemaps()
    df.to_csv(out, index=False)
    print(f"[acoustic] wrote {len(df)} rows -> {out}")
