"""Download and standardize the DementiaNet audio.

Data source: https://github.com/shreyasgite/dementianet (MIT license)
  - Metadata + YouTube URLs: Google Sheet (export to data/external/)
  - Audio clips: Google Drive folders (dementia / no-dementia)

This module is a SCAFFOLD. Fill in the Google Drive folder IDs from the repo
README, then run. Audio lands in data/raw/ (gitignored) and standardized
16 kHz mono wavs in data/interim/.
"""
from __future__ import annotations

import subprocess

from dementianet.utils import load_config, resolve

# TODO: paste the Google Drive folder IDs from the DementiaNet README links.
GDRIVE_FOLDERS = {
    "dementia": "PASTE_DEMENTIA_FOLDER_ID",
    "nodementia": "PASTE_NODEMENTIA_FOLDER_ID",
}


def download_clips() -> None:
    """Download raw audio folders from Google Drive using gdown."""
    cfg = load_config()
    raw = resolve(cfg["paths"]["data_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    for label, folder_id in GDRIVE_FOLDERS.items():
        out = raw / label
        print(f"[download] {label} -> {out}")
        subprocess.run(
            ["gdown", "--folder", folder_id, "-O", str(out)],
            check=True,
        )


def standardize_audio() -> None:
    """Convert every raw clip to 16 kHz mono WAV in data/interim/ via ffmpeg."""
    cfg = load_config()
    raw = resolve(cfg["paths"]["data_raw"])
    interim = resolve(cfg["paths"]["data_interim"])
    sr = cfg["audio"]["target_sample_rate"]
    interim.mkdir(parents=True, exist_ok=True)

    for src in raw.rglob("*"):
        if src.suffix.lower() not in {".mp3", ".m4a", ".wav", ".flac"}:
            continue
        rel = src.relative_to(raw).with_suffix(".wav")
        dst = interim / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), str(dst)],
            check=True,
        )
        print(f"[standardize] {src.name} -> {dst}")


if __name__ == "__main__":
    download_clips()
    standardize_audio()
