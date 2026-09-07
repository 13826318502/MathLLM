# Correction Round 11 training

- Base model: models/Qwen2.5-7B-Instruct-modelscope.
- Data: data/processed/round-10-staging/train.json and eval.json.
- QLoRA 4-bit with assistant-only loss.
- One epoch, batch size 1, gradient accumulation 16.
- Learning rate 5e-5, 10% warmup, cosine scheduler.
- Output: outputs/correction-round-11.
- Select the checkpoint with the lowest eval_loss before merging.
