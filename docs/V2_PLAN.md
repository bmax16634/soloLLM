# SoloLLM V2 Plan

This is the execution plan for turning SoloLLM from a good portfolio project into a standout one. The goal is not just "make the model bigger." The goal is to show a complete engineering loop: baseline, diagnosis, improved architecture, reproducible training, fair evaluation, and a polished public demo.

## North Star

SoloLLM v2 should tell this story:

> I built a GPT-style language model from scratch, published a working v1 baseline, then rebuilt the system into a cleaner v2 with better architecture, reproducible training, stronger evaluation, and a public demo that makes the improvement visible.

## 10/10 Scorecard

| Area | 10/10 Standard | Current State | Gap |
| --- | --- | --- | --- |
| Repository clarity | A reviewer can understand the project in 2 minutes and run a smoke test in 5 minutes. | README and layout are now cleaned up. | Add tests, toy data, exact commands, and screenshots. |
| Reproducibility | CPU smoke tests and short toy training run work without private data or local paths. | Config paths are cleaner, but no formal tests yet. | Add fixtures, tests, and dry-run commands. |
| Architecture | v2 has clear technical improvements over v1, not only larger hyperparameters. | RoPE attention exists. | Add SDPA, tied embeddings, sequence checks, parameter counts, and model summary. |
| Training system | Runs are resumable, logged, and easy to compare. | JSONL metrics and checkpoint metadata exist. | Add resume, scheduler, seeds, run summaries, and plots. |
| Evaluation | v1, v2, and GPT-2 are compared on the same fixed held-out split. | Eval exists but needs standardization. | Add fixed eval config, result table, and repeatable command. |
| Demo | Public demo works without manual local checkpoint setup. | Streamlit app exists. | Load checkpoint from Hugging Face or provide a clear fallback. |
| Portfolio story | Shows problem, constraints, architecture, results, tradeoffs, and lessons learned. | README has the new framing. | Add project brief, figures, result table, and model cards. |

## Final Deliverables

These are the artifacts that make v2 feel finished:

- `tests/test_models.py`: v1/v2 forward-pass and shape tests.
- `tests/test_training_smoke.py`: tiny CPU training smoke test on a fixture shard.
- `configs/` or test fixture config for reproducible local checks.
- `scripts/train_v2_toy.py` or documented dry-run command.
- `eval/eval.py` refactored to accept CLI args for checkpoint, shard range, model type, and output path.
- `docs/PROJECT_BRIEF.md`: portfolio writeup with architecture, constraints, results, and lessons.
- `docs/results/v1_vs_v2.md`: fixed prompts, perplexity table, hardware, training tokens, and screenshots/plots.
- Hugging Face model card for any published v2 checkpoint.
- Streamlit demo that can download/load published weights or clearly guide the user when weights are missing.

## Milestone 1: Reproducibility Foundation

Goal: make the project easy to verify without full training data.

Tasks:

- Add `pytest` and lightweight tests.
- Add a tiny token shard fixture under `tests/fixtures/`.
- Add smoke tests that instantiate `SoloGPT_v1` and `SoloGPT_v2`, run forward passes, and verify logits shape.
- Add a toy v2 training check that runs 2-3 optimizer steps on CPU.
- Make training scripts fail early with clear errors when shards or checkpoints are missing.
- Add documented commands to README:
  - `python -m pytest`
  - `python -m sologpt_v2.pretrain --dry-run` if we add CLI flags, or an equivalent toy script.

Acceptance criteria:

- A clean clone can run tests without OpenWebText or model weights.
- No test depends on `/home/bmx`, private data, local checkpoints, or a GPU.

## Milestone 2: V2 Architecture Upgrade

Goal: make v2 technically defensible as an improvement over v1.

Tasks:

- Replace manual attention math with `torch.nn.functional.scaled_dot_product_attention` when available.
- Keep a readable fallback if needed.
- Add tied input/output embeddings if compatible with the current model shape.
- Add explicit sequence length checks before attention/RoPE usage.
- Add parameter count reporting at startup.
- Add model summary fields to run metadata:
  - parameter count
  - context length
  - layers
  - heads
  - embedding width
  - vocab size
- Decide whether to keep LayerNorm or move to RMSNorm/pre-norm after a small ablation.

Acceptance criteria:

- `sologpt_v2/model.py` reads like a deliberate GPT implementation.
- The README can list v2 improvements in specific terms: RoPE, SDPA, tied embeddings if used, better metadata, larger context, better training loop.

## Milestone 3: Training System Upgrade

Goal: make training runs resumable, logged, and comparable.

Tasks:

- Add CLI arguments to v2 training:
  - `--config`
  - `--shard-dir`
  - `--output-dir`
  - `--resume`
  - `--max-steps`
  - `--max-tokens`
  - `--dry-run`
- Add deterministic seed handling for Python, NumPy, PyTorch, CUDA, and dataloader workers.
- Add resume-from-checkpoint for:
  - model state
  - optimizer state
  - scaler state
  - global step
  - token counters
  - shard index if practical
- Add warmup plus cosine learning-rate schedule.
- Save `metrics_summary.json` at the end of every run.
- Add a small plotting script to turn JSONL metrics into a training curve PNG.

Acceptance criteria:

- Interrupting and resuming training does not lose the run.
- Every training run has enough metadata to compare it later.
- The outputs folder contains structured artifacts, not just checkpoint blobs.

## Milestone 4: Fair Evaluation

Goal: prove what improved, and be honest about what did not.

Tasks:

- Refactor eval into a CLI:
  - `--model v1|v2|gpt2`
  - `--checkpoint`
  - `--shard-dir`
  - `--start-shard`
  - `--end-shard`
  - `--output-json`
- Fix one held-out shard range for the portfolio comparison.
- Evaluate:
  - SoloGPT v1 checkpoint
  - SoloGPT v2 checkpoint
  - GPT-2 small baseline
- Report:
  - perplexity
  - parameter count
  - training tokens
  - context length
  - hardware
  - train time
  - checkpoint size
- Add a short limitation note if v2 is not better on every metric.

Acceptance criteria:

- The results are reproducible from one documented command.
- The comparison is fair enough that an ML reviewer would not immediately dismiss it.

## Milestone 5: Demo And Publishing

Goal: make the public-facing project work without hidden local state.

Tasks:

- Publish the best v2 checkpoint to Hugging Face if the quality is worth showing.
- Add model cards for v1 and v2.
- Update Streamlit loading:
  - use local checkpoint if present
  - otherwise download from Hugging Face or show a precise setup message
- Add a small prompt gallery:
  - factual question
  - creative completion
  - instruction-style answer
  - failure case
- Capture screenshots or short GIFs for the README/project brief.

Acceptance criteria:

- The demo can be run by someone other than the author.
- The README makes clear what checkpoint the demo uses.

## Milestone 6: Portfolio Packaging

Goal: make the project easy to judge as an engineering accomplishment.

Tasks:

- Add `docs/PROJECT_BRIEF.md` with:
  - problem
  - constraints
  - architecture
  - data pipeline
  - training loop
  - evaluation
  - v1 to v2 improvements
  - lessons learned
- Add a compact results table to the README.
- Add architecture diagram or block diagram.
- Add training curve plot.
- Tag release points:
  - `v1-clean`
  - `v2-alpha`
  - `v2-portfolio`

Acceptance criteria:

- A portfolio visitor can understand the project without reading all source code.
- A technical reviewer can still find enough detail to trust the work.

## Execution Order

1. Add tests and toy fixtures.
2. Add CLI/dry-run support to v2 training.
3. Upgrade v2 architecture internals.
4. Add resume, seeds, scheduler, and summaries.
5. Refactor eval into a CLI.
6. Run a controlled v1/v2/GPT-2 comparison.
7. Publish or document v2 checkpoint.
8. Add project brief, plots, screenshots, and final README results.
9. Merge the finished portfolio state to `main`.
10. Tag the final v2 state.

## Branch Strategy

- `experimental`: current cleanup and planning branch.
- `main`: polished public branch after cleanup is reviewed.
- `v2-dev`: implementation branch for tests, CLI, architecture, and training improvements.
- `v2-training`: long-running training branch or local run branch; merge only final code, metrics, and docs.

## Definition Of Done

SoloLLM v2 is portfolio-ready when all of these are true:

- `python -m pytest` passes from a clean clone.
- README has exact setup, smoke test, training, evaluation, and demo commands.
- v2 has a documented architecture improvement list.
- v2 has a fixed evaluation result against v1 and GPT-2.
- Training metrics and at least one curve are published.
- Checkpoints are handled through Hugging Face or documented local setup.
- The repo has no personal absolute paths, stale imports, empty notebooks, or committed generated artifacts.

## Honest Rating Path

Current cleaned project: 7/10.

After tests, reproducible toy run, and polished README: 8/10.

After real v1/v2/GPT-2 evaluation and public demo weights: 9/10.

After project brief, plots, model cards, tags, and a clean v2 checkpoint story: 10/10.
