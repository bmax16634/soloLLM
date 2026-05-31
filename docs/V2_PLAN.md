# SoloLLM V2 Plan

This plan turns the experimental v2 code into a second portfolio iteration that can be compared directly against the published v1 baseline.

## Objective

Build a clearly improved SoloGPT iteration with better architecture, more reliable training, and a cleaner public story. The v2 deliverable should be easy to explain in a portfolio as: "I trained a baseline model from scratch, identified the engineering gaps, then built a better second iteration."

## Current V2 Baseline

Already present on the cleanup branch:

- `sologpt_v2/model.py` implements causal self-attention with RoPE.
- `sologpt_v2/pretrain.py` writes run metadata, JSONL metrics, checkpoint events, validation events, and training summaries.
- `sologpt_v2/config.json` separates the v2 hyperparameters from v1.
- The training path is now config-driven instead of hardcoded to one local machine.

## Milestone 1: Make V2 Reproducible

- Add a small smoke test that instantiates v1 and v2, runs a forward pass, and verifies logits shape.
- Add a tiny fixture config for CPU-only validation.
- Add a documented command for a short dry run on a toy shard.
- Ensure checkpoints, metrics, and summaries all land under `outputs/sologpt_v2/`.

Success signal: a reviewer can run a lightweight v2 sanity check without downloading OpenWebText or training the full model.

## Milestone 2: Improve The Model

- Switch attention to `torch.nn.functional.scaled_dot_product_attention` when available.
- Tie token embedding and LM head weights if the final architecture supports it cleanly.
- Add explicit max sequence length checks before positional or RoPE cache usage.
- Consider RMSNorm or pre-norm block ordering after validating against the current implementation.
- Add parameter-count reporting in the training startup logs.

Success signal: v2 has concrete architectural improvements over v1 beyond just being larger.

## Milestone 3: Improve Training Quality

- Add resume-from-checkpoint support for model, optimizer, scaler, token counters, and run metadata.
- Add a learning-rate schedule with warmup and cosine decay.
- Add deterministic seed handling for torch, numpy, and dataloader workers.
- Add gradient accumulation accounting based on optimizer steps, not just micro-batches.
- Save a compact `metrics_summary.json` for portfolio plots.

Success signal: interrupted training can resume cleanly, and training curves are easy to publish.

## Milestone 4: Evaluate Against V1

- Standardize an eval split and document the exact shard range.
- Report perplexity for v1, v2, and GPT-2 small on the same held-out data.
- Add generation examples with the same prompts across v1 and v2.
- Include training tokens, parameter count, context length, hardware, and wall-clock time in the results table.

Success signal: the portfolio can show a fair before/after comparison instead of only describing implementation changes.

## Milestone 5: Portfolio Packaging

- Add a short project brief with problem, constraints, architecture, training loop, results, and lessons learned.
- Add model cards for published checkpoints.
- Update the Streamlit demo to download a checkpoint from Hugging Face when a local checkpoint is missing.
- Tag the cleaned v1 state and the completed v2 state.

Success signal: the repository reads like a finished engineering project, not a training scratchpad.

## Suggested Branch Flow

1. Keep cleanup work on `experimental` until imports, docs, and smoke checks pass.
2. Merge `experimental` into `main` as the portfolio-ready v1/v2 scaffold.
3. Create a dedicated `v2-training` branch for longer training runs and experiment logs.
4. Merge only reproducible code, docs, and final metrics back into `main`.

## Portfolio Story

Use v1 as the baseline:

- Built a GPT-style model from scratch.
- Built tokenization, pretraining, evaluation, generation, and demo paths.
- Published weights externally.

Use v2 as the improvement:

- Reworked the package layout and documentation.
- Added RoPE and better training instrumentation.
- Added checkpoint metadata and structured metrics.
- Planned fair evaluation against v1 and GPT-2.
