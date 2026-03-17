# LoRA fine-tuning setup in OpenCLIP

## Current state

**LoRA is supported via PEFT.** Install with:

```bash
pip install peft
# or from source:
pip install git+https://github.com/huggingface/peft.git@main
```

Training applies LoRA **after** model creation and any bitsandbytes replacement, **before** `lock_image` / `lock_text` and DDP. When `--lora` is set, image/text locking is skipped so adapter parameters stay trainable. `logit_scale` and `logit_bias` remain trainable.

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora` | False | Enable LoRA fine-tuning (PEFT). |
| `--lora-r` | 8 | LoRA rank. |
| `--lora-alpha` | 32 | LoRA alpha scaling. |
| `--lora-dropout` | 0.0 | LoRA dropout. |
| `--lora-target-modules` | out_proj c_fc c_proj | Module name substrings to adapt (default fits CLIP/ViT in this codebase). |

For timm-based vision towers you may need different targets, e.g. `--lora-target-modules qkv proj fc1 fc2`.

---

## Where LoRA hooks in

In [main.py](src/open_clip_train/main.py):

1. `create_model_and_transforms(...)` → model, preprocess
2. Optional: `use_bnb_linear` → replace_linear(model, ...)
3. **If `--lora`:** `get_peft_model(model, LoraConfig(...))`, then set `logit_scale` / `logit_bias` to trainable and `model.print_trainable_parameters()`
4. Optional: `lock_image` / `lock_text` (skipped when `--lora`)
5. Optional: grad_checkpointing, then DDP, optimizer, train/eval, checkpointing

Checkpoints are the full `state_dict()` (base + LoRA). To resume, use the same `--lora` (and same target modules) so the loaded state dict matches the PEFT model.

---

## Summary

| Item | Status |
|------|--------|
| LoRA in codebase | **Implemented** via PEFT. |
| Insertion point | After model creation and bnb, before lock_* and DDP. |
| Trainable params | LoRA adapters + `logit_scale` / `logit_bias`. |
| Lock image/text | Skipped when `--lora`. |
| DDP (multi-GPU) | `find_unused_parameters=True` is set when using LoRA so frozen base params do not trigger DDP reducer errors. |
| Checkpointing | Full state dict; load with same LoRA config to resume. |

To run with LoRA:

```bash
python -m open_clip_train.main --lora --lora-r 8 --lora-alpha 32 \
  --model ViT-B-16-SigLIP-256 --pretrained ... --train-data ... # etc.
```

---

## Multi-GPU (DDP)

LoRA works with distributed data parallel (DDP). Use `torchrun` and pass the same CLI args; `init_distributed_device` will set up DDP from `RANK` / `LOCAL_RANK` / `WORLD_SIZE`.

```bash
# 2 GPUs (default)
./scripts/run_tiny_multigpu_test.sh

# 4 GPUs
NGPUS=4 ./scripts/run_tiny_multigpu_test.sh

# Or manually
torchrun --nproc_per_node=2 -m open_clip_train.main --lora --train-data ... # etc.
```

The script `scripts/run_tiny_multigpu_test.sh` runs a short DDP + LoRA + masked-attr test (1 epoch, tiny data) and requires at least 2 GPUs. It sets `NCCL_P2P_DISABLE=1` by default to avoid hangs on systems where GPU-to-GPU P2P fails (override with `NCCL_P2P_DISABLE=0` if your cluster supports P2P). For manual `torchrun` runs on such systems, set the same before launching: `export NCCL_P2P_DISABLE=1`.
