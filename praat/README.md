# Praat scripts (uhm-o-meter) — download separately

This folder holds third-party Praat scripts used to detect **filled pauses**
(uh/um) acoustically. They are **not committed** to the repo (see `.gitignore`);
download them yourself and drop them here.

## What to download

From the uhm-o-meter site (De Jong, Pacilly & Heeren, 2021):
https://sites.google.com/view/uhm-o-meter/scripts

Save into this folder with these exact names (so the config paths match):

```
praat/
├── filledpauses.praat      <- the "Detect Filled Pauses" script
└── (any companion script it @includes, e.g. the syllable-nuclei script)
```

If `filledpauses.praat` references another script via `include` or `runScript`,
download that too and keep it in this same folder.

## How it's used

1. Run the script over `data/interim/` (the 16 kHz mono clips). Two options:
   - **Praat GUI (most reliable):** open the script, Run, point it at
     `data/interim/`, set language = English, threshold = 1.2. Save the results
     table to `data/interim/filledpauses_output.txt`.
   - **Headless:** `praat --run praat/filledpauses.praat <args>` — argument order
     depends on the script's `form`; confirm by opening the script.
2. `python -m dementianet.features.filled_pauses` parses that output into
   `data/processed/filled_pause_features.csv`.

## Attribution (cite in your write-up)

De Jong, N.H., Pacilly, J., & Heeren, W. (2021). *PRAAT scripts to measure speed
fluency and breakdown fluency in speech automatically.* Assessment in Education:
Principles, Policy & Practice, 28(4), 456–476.

## Note on language

The filled-pause detector is tuned for English (and Dutch). Most DementiaNet
speakers are English; flag any non-English speakers as a limitation, since
detection reliability drops for other languages.
