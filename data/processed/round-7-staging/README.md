# Round 7 staging data

- `train.json`: 215 records = 150 new `correction-round-7` records + 65 ordinary replay records.
- Correction ratio: 69.77%; replay ratio: 30.23%.
- `eval.json`: 40 ordinary holdout records; Round 7 corrections remain train-only.
- Historical `correction-round-4/5/6` sources are excluded.
- `data/eval/test.json` and `data/eval/regression/regression.json` are protected and never copied into this staging split.
- `manifest.json` records the counts, seed, source and subject distributions.

This staging set is for the single Round 7 run, whose base model is the Round 6
merged model. It is not a replacement for the independent test or regression
sets used after merging.
