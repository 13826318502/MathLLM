# Round 16 full-base reset

- Base: original `Qwen2.5-7B-Instruct`.
- Train data: all clean original data plus 180 new verified correction examples.
- Independent correction validation: 60 examples.
- Original test and historical regression sets remain frozen and excluded.
- Learning rate `5e-5`, warmup `15` steps, cosine decay, one epoch, assistant-only loss.
- Before training, run `python scripts/build_correction_round16.py` and inspect `manifest.json`.
