from pathlib import Path

import pandas as pd

rows = []
for wav in Path("data/raw").rglob("*.wav"):
    label, speaker = wav.parts[2], wav.parts[3]  # data/raw/<label>/<speaker>/<file>.wav
    rows.append({"clip_path": wav.name, "speaker_id": speaker, "label": label})
pd.DataFrame(rows).to_csv("data/processed/manifest.csv", index=False)
