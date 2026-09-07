# Round 15 reset training

- Base model: the original `Qwen2.5-7B-Instruct`, not a previously merged model.
- Train mix: 420 clean original examples and 180 newly verified correction examples.
- Independent correction validation: 60 examples.
- The frozen original test set and historical regression set are excluded.
- Conservative settings: one epoch, learning rate `2e-5`, cosine decay, and assistant-only loss.
- Before uploading, run `python scripts/build_correction_round15.py` and inspect `manifest.json`.
