"""Pause / silence feature extraction -- the project's scientific core.

Implements the pause markers highlighted in the report:
  - pause count and total/percentage silence duration (PSD)
  - short vs long pauses split at ~180 ms (He et al. 2025 bimodal threshold)
  - mean / std pause duration, articulation rate

These are extracted from the audio envelope; no transcript required, which
suits the noisy YouTube-sourced DementiaNet clips.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from dementianet.utils import load_config, resolve


def pause_features(wav_path: Path, min_pause_ms: int, silence_db: float) -> dict:
    """Compute pause statistics for a single clip."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration_s = len(y) / sr

    # Non-silent intervals (samples) using an energy threshold.
    intervals = librosa.effects.split(y, top_db=abs(silence_db))
    if len(intervals) == 0:
        return {"clip_path": str(wav_path.name), "valid": False}

    # Silence gaps = spaces between voiced intervals.
    gaps_s = []
    for (_s0, e0), (s1, _e1) in zip(intervals[:-1], intervals[1:], strict=False):
        gaps_s.append((s1 - e0) / sr)
    gaps_s = np.array(gaps_s)

    voiced_s = sum((e - s) for s, e in intervals) / sr
    silence_s = duration_s - voiced_s
    min_pause_s = min_pause_ms / 1000.0
    long_pauses = gaps_s[gaps_s >= min_pause_s]
    short_pauses = gaps_s[gaps_s < min_pause_s]

    dur = duration_s if duration_s else np.nan
    return {
        "clip_path": str(wav_path.name),
        "valid": True,
        "duration_s": duration_s,
        "psd": silence_s / dur,  # % silence duration
        # Raw counts (duration-confounded; kept for reference).
        "n_pauses": len(gaps_s),
        "n_long_pauses": int(len(long_pauses)),
        "n_short_pauses": int(len(short_pauses)),
        # Per-second rates (duration-normalized; use these to separate
        # "pauses because impaired" from "pauses because the clip is long").
        "pause_rate": len(gaps_s) / dur,
        "long_pause_rate": len(long_pauses) / dur,
        "short_pause_rate": len(short_pauses) / dur,
        "mean_pause_s": float(np.mean(gaps_s)) if len(gaps_s) else 0.0,
        "std_pause_s": float(np.std(gaps_s)) if len(gaps_s) else 0.0,
        "long_pause_ratio": len(long_pauses) / len(gaps_s) if len(gaps_s) else 0.0,
        "articulation_rate": len(intervals) / voiced_s if voiced_s else np.nan,
    }


def extract_all() -> pd.DataFrame:
    cfg = load_config()
    interim = resolve(cfg["paths"]["data_interim"])
    min_pause_ms = cfg["features"]["pause"]["min_pause_ms"]
    silence_db = cfg["features"]["pause"]["silence_threshold_db"]

    rows = [
        pause_features(wav, min_pause_ms, silence_db)
        for wav in sorted(Path(interim).rglob("*.wav"))
    ]
    if not rows:
        raise FileNotFoundError(f"No wavs in {interim}. Run download.py first.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()
    out = resolve(cfg["paths"]["data_processed"]) / "pause_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = extract_all()
    df.to_csv(out, index=False)
    print(f"[pauses] wrote {len(df)} rows -> {out}")
