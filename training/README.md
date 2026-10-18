# Training and Prediction

This directory contains the LLaMA-Factory configurations used for training and
prediction with Qwen3.5-4B.

Run supervised fine-tuning:

```bash
bash training/run_sft.sh
```

Run prediction on the test set:

```bash
bash training/run_predict.sh
```

The training configuration is stored in
`training/configs/qwen3_5_4b_lora_sft.yaml`. The prediction configuration is
stored in `training/configs/qwen3_5_4b_lora_predict.yaml`.

Expected LLaMA-Factory dataset names:

- `gex_train`
- `gex_test`

The fine-tuned LoRA adapter is written to `saves/Qwen3.5-4B/lora/4B_sft`, and
prediction outputs are written under `saves/Qwen3.5-4B/lora/eval_of_4B_sft`.
