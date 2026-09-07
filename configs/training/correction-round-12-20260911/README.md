# Correction Round 12

- Base model: `outputs/correction-round-10-merged`
- Training data: `data/processed/round-12-staging/train.json`
- Mix: 360 reviewed correction records + 154 stratified ordinary replay records (about 70/30)
- Internal eval: independent `data/eval/correction-validation/round-10.json`
- Test and regression remain protected and are only used after merging.
- This is a controlled replay run because no new correction source has been added yet.
