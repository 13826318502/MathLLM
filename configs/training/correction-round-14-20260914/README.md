# Correction Round 14

- Base model: `outputs/correction-round-10-merged`
- Training mix: targeted correction data for regression failures plus ordinary replay at 70/30.
- Independent correction validation: `data/eval/correction-validation/round-14.json`
- The original test set and historical regression set remain protected.
- The 80% accuracy target is an evaluation target, not a guaranteed result.
