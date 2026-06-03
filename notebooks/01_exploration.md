# Notebook 01 — Exploratory Data Analysis

> Convert to `.ipynb` with `jupytext` or just start a notebook here. Kept as
> markdown so it diffs cleanly in git. Notebooks are for *exploration*; once
> something works, move it into `src/`.

## Goals
1. Load `data/external/dementianet_metadata.csv` and sanity-check class balance
   (dementia vs no-dementia) and counts per time-before-diagnosis bucket.
2. Inspect clip durations and sample rates after standardization.
3. **Confound audit (important):** compare audio quality proxies (SNR, clip
   length, recording year if available) *across classes*. If dementia clips are
   systematically older/noisier, the model can cheat — document this.
4. Plot distributions of key pause features (PSD, long-pause ratio) by class and
   by time bucket. This previews the core scientific result.
