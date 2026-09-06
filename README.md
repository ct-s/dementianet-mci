# DementiaNet — Speech-Based Early Detection of Cognitive Decline

A master's mini-project (computational neuroscience). **Research question:**
*How many years before clinical diagnosis does a detectable speech signal emerge,
and how much of it is carried by pause structure alone?*

Built on the [DementiaNet](https://github.com/shreyasgite/dementianet) corpus
(MIT-licensed, openly downloadable): ~100 public figures with confirmed dementia
+ ~100 cognitively healthy individuals aged 85+, with longitudinal spontaneous
speech sampled from after diagnosis back to 10–15 years before symptom onset.

## Project structure

```
dementianet-mci/
├── config/config.yaml        # all paths, hyperparameters, constants
├── data/                     # gitignored (see Data section)
│   ├── raw/                  # downloaded audio
│   ├── interim/              # standardized 16 kHz mono wavs
│   ├── processed/            # feature tables (CSV)
│   └── external/             # metadata CSV from the DementiaNet Google Sheet
├── notebooks/                # exploration (markdown/jupytext)
├── src/dementianet/          # importable package
│   ├── data/download.py      # download + standardize audio
│   ├── features/pauses.py    # pause/silence features (scientific core)
│   ├── features/acoustic.py  # openSMILE eGeMAPS (jitter, shimmer, HNR, F0)
│   ├── models/train.py       # speaker-independent CV baselines
│   └── utils.py              # config loading, seeding
├── tests/                    # pytest smoke tests
├── results/{figures,tables}/ # gitignored outputs
├── scripts/run_pipeline.sh   # end-to-end runner
├── environment.yml           # conda environment
└── pyproject.toml            # package metadata + ruff/pytest config
```

## Setup

### 1. Create the environment (conda / miniforge)
```bash
conda env create -f environment.yml
conda activate dementianet
pip install -e .          # install the src package in editable mode
pre-commit install        # enable formatting/lint hooks on commit
```

### 2. Get the data
The data is **not** in this repo (audio is gitignored). Download from the
[DementiaNet repo](https://github.com/shreyasgite/dementianet):
- Save the **Google Sheet** (URLs + metadata) as
  `data/external/dementianet_metadata.csv`.
- Copy the Google Drive **folder IDs** for the dementia / no-dementia clips into
  `src/dementianet/data/download.py` (`GDRIVE_FOLDERS`).

Then run the pipeline:
```bash
bash scripts/run_pipeline.sh
```

## License
Code: MIT (see `LICENSE`). Data: governed by DementiaNet's terms — research use only.
