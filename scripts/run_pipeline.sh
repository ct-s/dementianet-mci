#!/usr/bin/env bash
# End-to-end pipeline. Run from the repo root after `conda activate dementianet`.
set -euo pipefail

echo "==> 1/4 Download + standardize audio"
python -m dementianet.data.download

echo "==> 2/4 Extract pause features"
python -m dementianet.features.pauses

echo "==> 3/4 Extract acoustic (eGeMAPS) features"
python -m dementianet.features.acoustic

echo "==> 4/4 Train + evaluate baselines"
python -m dementianet.models.train

echo "Done. See results/tables/baseline_auc.csv"
