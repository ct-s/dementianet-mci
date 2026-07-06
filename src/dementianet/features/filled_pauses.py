"""Filled-pause + fluency features via Praat "Syllable Nuclei v3".

v3 (de Jong, Wempe & Pacilly 2019) detects syllable nuclei and, with
Detect_Filled_Pauses on, internally runs FilledPauses.praat and reports nrFP
(filled-pause count) and tFP (filled-pause time). We drive v3 on each clip as a
selected Sound object and read its one-row Info output:

  name, nsyll, npause, dur(s), phonationtime(s), speechrate, articulation_rate,
  ASD, nrFP, tFP(s)

Derived features (the FILLED block): filled-pause count, rate (per second),
filled-to-silent ratio, and filled-pause time ratio.

Run:
  python -m dementianet.features.filled_pauses          # probe the first clip
  python -m dementianet.features.filled_pauses --all     # process every clip

Needs praat/syllablenuclei_v3 and praat/filledpauses.praat (see praat/README.md).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dementianet.utils import load_config, resolve

# Numeric columns of the v3 output row, in order (after the name field).
V3_FIELDS = [
    "nsyll",
    "npause",
    "dur",
    "phonationtime",
    "speechrate",
    "articulation_rate",
    "asd",
    "nrFP",
    "tFP",
]


def _ensure_filledpauses_sibling(v3_path: Path) -> None:
    """v3 calls `runScript: "FilledPauses.praat"` (exact case). Make sure a file
    with that name sits next to v3 (copy from the lowercase download if needed)."""
    target = v3_path.parent / "FilledPauses.praat"
    src = v3_path.parent / "filledpauses.praat"
    if not target.exists() and src.exists():
        shutil.copy(src, target)


def _parse_row(text: str, stem: str) -> dict | None:
    """Extract the 9 numeric fields from v3's Info output for one clip.

    Name-agnostic: v3 may splice F0 "Warning:" lines into the row (referencing
    whatever the Sound is named, e.g. 'untitled'). We cut each line at 'Warning:'
    so the numeric data survives regardless of the object name.
    """
    lines = []
    for ln in text.splitlines():
        ln = ln.split("Warning:")[0].strip()  # drop any warning fragment
        if not ln or ln.lower().startswith("name,"):
            continue
        lines.append(ln)
    fields = [f.strip() for f in " ".join(lines).split(",") if f.strip()]
    if len(fields) < 10:
        return None
    try:
        nums = [float(x) for x in fields[1:10]]
    except ValueError:
        return None
    return dict(zip(V3_FIELDS, nums, strict=False))


def run_v3(limit: int | None = None, dump: bool = True) -> pd.DataFrame:
    """Run v3 on each interim clip; return a dataframe of raw v3 fields + clip_path."""
    import parselmouth

    cfg = load_config()
    v3 = resolve(cfg["paths"]["syllable_nuclei_script"])
    if not v3.exists():
        raise FileNotFoundError(f"{v3} not found (see praat/README.md).")
    _ensure_filledpauses_sibling(v3)

    fp = cfg["features"]["filled_pauses"]
    language = "English" if str(fp.get("language", "English")).lower().startswith("en") else "Dutch"
    # form order: FileSpec, Pre_processing, Silence_thr, Min_dip, Min_pause,
    # Detect_Filled_Pauses, Language, FP_threshold, Data, DataCollectionType, Keep_Objects
    args = [
        "",
        "None",
        float(fp.get("silence_threshold_db", -25)),
        float(fp.get("min_dip_db", 2)),
        float(fp.get("min_pause_s", 0.3)),
        True,
        language,
        float(fp.get("threshold", 1.0)),
        "Praat Info window",
        "OverWriteData",
        False,
    ]

    interim = resolve(cfg["paths"]["data_interim"])
    wavs = sorted(Path(interim).rglob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    if not wavs:
        raise FileNotFoundError(f"No wavs in {interim}.")

    rows = []
    for i, wav in enumerate(wavs):
        snd = parselmouth.Sound(str(wav))
        out = parselmouth.praat.run_file([snd], str(v3), *args, capture_output=True)
        text = out[1] if isinstance(out, tuple) else str(out)
        if i == 0 and dump:
            print("---- raw v3 output for first clip ----")
            print(text)
            print("--------------------------------------")
        rec = _parse_row(text, wav.stem)
        if rec is None:
            print(f"[warn] could not parse output for {wav.name}")
            continue
        rec["clip_path"] = wav.name
        rows.append(rec)

    return pd.DataFrame(rows)


def build_features(df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Derive the FILLED feature block and write filled_pause_features.csv."""
    cfg = load_config()
    df = run_v3() if df_raw is None else df_raw

    out = pd.DataFrame({"clip_path": df["clip_path"]})
    dur = df["dur"].replace(0, np.nan)
    out["n_filled_pauses"] = df["nrFP"]
    out["filled_pause_rate"] = df["nrFP"] / dur
    out["filled_to_silent_ratio"] = df["nrFP"] / df["npause"].replace(0, np.nan)
    out["filled_pause_time_ratio"] = df["tFP"] / dur

    path = resolve(cfg["paths"]["data_processed"]) / "filled_pause_features.csv"
    out.to_csv(path, index=False)
    print(f"[filled_pauses] wrote {len(out)} rows -> {path}")
    print(out.describe().round(3).to_string())
    return out


if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        build_features(run_v3())
    else:
        df = run_v3(limit=1)
        print("\n=== parsed record ===")
        parsed = df.to_string(index=False) if len(df) else "(parse failed -- see raw output above)"
        print(parsed)
        print("\nProbe done. If the row parsed, run with --all to process every clip.")
