# V5 Small-Model Research Plan

Date: July 19, 2026

## Frame

V5 is not another GPT-2-chasing phase.

The v4 closeout showed that SoloLLM can train a GPT-2-small-beating base model
from scratch at 151.9M parameters, and can get a slightly smaller-than-GPT-2
model to win the fixed multiple-choice suite. Continuing to tune against the
same GPT-2 comparison risks benchmark-chasing.

V5 reframes the project around small-model research on local hardware:

> Build and evaluate the best small base models this project can train and run
> on one RTX 3090, measured by quality, size, speed, memory, quantized behavior,
> and practical usefulness.

## Research Questions

1. Which changes most improve small base models under a single-3090 budget:
   data quality, data quantity, architecture shape, tokenizer/context, or
   training recipe?
2. How close can a from-scratch SoloLLM model get to modern small baselines such
   as SmolLM2 135M/360M and Pythia 160M/410M?
3. What model size is the best local Pareto point for this hardware: 150M,
   250M-300M, or 350M-450M?
4. How much quality survives quantization and local inference constraints?

## Non-Goals

- Do not optimize only for GPT-2.
- Do not claim assistant quality from base-LM metrics.
- Do not train another long run before the v5 eval frame is frozen.
- Do not change architecture, tokenizer, context length, data mix, and optimizer
  all in one experiment.
- Do not treat one benchmark table as proof that a model is broadly better.

## Baselines

Primary comparison set:

| Model | Params | Role |
| --- | ---: | --- |
| SoloLLM v4.2 150M | 151.87M | Current best SoloLLM quantitative checkpoint |
| SoloLLM v4.2 123M | 123.55M | Smaller-than-GPT-2 ablation |
| GPT-2 small | 124.44M | Historical GPT-2-class reference |
| Pythia 160M | 160M | Controlled open baseline near SoloLLM scale |
| Pythia 410M | 410M | Larger controlled open baseline |
| SmolLM2 135M | 135M | Modern compact baseline near SoloLLM scale |
| SmolLM2 360M | 360M | Modern compact upper-bound baseline |
| TinyLlama 1.1B | 1.1B | Stretch local baseline, not same-size fair comparison |

Reference links:

- SmolLM2 135M model card: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
- Pythia 160M model card: <https://huggingface.co/EleutherAI/pythia-160m>
- LM Evaluation Harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- TinyLlama 1.1B model card: <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>

## Eval Suite 2.0

V5 starts by broadening and freezing evaluation before more training.

### Tier 1: Continuity Metrics

These preserve comparability with v2/v3/v4:

- project held-out PPL on reserved shards `58:60`,
- WikiText-2 BPB/PPL,
- LAMBADA BPB/PPL,
- LAMBADA last-token accuracy,
- LAMBADA greedy exact-word accuracy,
- HellaSwag,
- PIQA,
- ARC-Easy,
- ARC-Challenge,
- WinoGrande,
- fixed-prompt generation metrics.

### Tier 2: Broader Standard Evals

Use lm-evaluation-harness where practical so results are less custom:

- OpenBookQA,
- SciQ,
- BoolQ,
- COPA,
- MMLU as a diagnostic only, not a target at this model scale,
- optional GSM8K only as a sanity check; small base models are expected to be
  weak without instruction tuning.

### Tier 3: Local Usefulness

Measure whether a model is actually good to run locally:

- tokens/sec at FP16/BF16,
- peak VRAM at inference,
- KV-cache growth with context length,
- CPU or partial-offload viability if supported,
- quantized model size and tokens/sec,
- quality after quantization on a small locked prompt set.

### Tier 4: Manual Prompt Review

Keep a fixed blind prompt set with short rubrics:

- factual continuation,
- explanation,
- narrative coherence,
- structured list,
- code-ish completion,
- reasoning-style continuation,
- repetition stress,
- refusal/safety-adjacent prompt.

Manual samples should be treated as qualitative evidence, not leaderboard
metrics.

## Anti-Overfitting Rules

- Split evals into `dev` and `lockbox`.
- Run `dev` often during proxies.
- Run `lockbox` only for final candidates.
- Do not change the training recipe based on lockbox failures more than once.
- Keep benchmark labels out of training data wherever feasible.
- Report failures and near-misses instead of hiding them.

## Model Directions

### Option A: 200M-300M Best Local Base Model

This is the most pragmatic first training target after Eval Suite 2.0.

- Reuse the proven v4.2 recipe and infrastructure.
- Increase capacity without jumping to a 1B-class run.
- Keep training feasible on one 3090.
- Compare against Pythia 160M and SmolLM2 135M first, then SmolLM2 360M.

Decision rule: train this if the goal is the best next SoloLLM model with
moderate risk.

### Option B: 350M-450M Modern Small Baseline Challenger

This is the more ambitious local-research target.

- Aim near Pythia 410M and SmolLM2 360M scale.
- Requires more careful memory planning, throughput testing, and possibly
  activation checkpointing or lower batch sizes.
- Needs more data than another 10B-token single-pass run to be worth it.

Decision rule: train this only after v5 eval/baseline infrastructure is working
and a smaller proxy shows the data recipe is strong.

### Option C: Tokenizer/Context Research

Defer this until the baseline board is done.

- A new tokenizer could help BPB and downstream metrics, but it breaks direct
  continuity with GPT-2-tokenizer runs.
- Longer context can help local usefulness, but it changes training cost and
  inference memory.

Decision rule: do not combine tokenizer/context changes with the first v5 model.

## First Milestone: V5 Eval Baseline Board

Before training a new model, build a baseline board with:

- SoloLLM v4.2 150M,
- SoloLLM v4.2 123M,
- GPT-2 small,
- Pythia 160M,
- Pythia 410M,
- SmolLM2 135M,
- SmolLM2 360M.

Output artifacts:

- `docs/results/v5_baseline_board.md`,
- raw JSON under `outputs/eval_suites/v5_baseline_board/`,
- one fixed prompt report,
- one inference/VRAM report.

Success criteria:

- every baseline runs end-to-end,
- metrics are comparable and documented,
- tokenizer differences are reported with BPB where appropriate,
- no new long training starts until this board exists.

### Baseline Board V0

The first quality-only board is complete:

- report: `docs/results/v5_baseline_board.md`,
- inference report: `docs/results/v5_inference_benchmark.md`,
- builder: `scripts/build_v5_baseline_board.py`,
- inference runner: `scripts/run_v5_inference_benchmark.py`,
- machine-readable output: `outputs/eval_suites/v5_baseline_board/baseline_board.json`.

This board reuses completed v3/v4 and modern-baseline eval artifacts. It is
enough to set direction, but it is not the final v5 board. Pending items are
Pythia 410M, TinyLlama 1.1B, quantized quality/speed, and optionally a cached
SoloLLM inference implementation.

## First Implementation Tasks

1. Create a v5 eval manifest listing models, tasks, metrics, and output paths.
2. Add or update a runner that evaluates local SoloLLM checkpoints and
   Hugging Face baselines in one repeatable suite.
3. Add a local inference benchmark script for tokens/sec, peak VRAM, and
   context-scaling behavior.
4. Freeze the v5 fixed prompt set and manual review rubric.
5. Run the baseline board.

## V5 Training Gate

Only train v5 after the baseline board is complete.

The first candidate should be either:

- a 200M-300M v4.2-style continuation with a better data strategy, or
- a proxy-backed 350M-450M run if the eval board shows the 360M-class baselines
  are the right target and the hardware budget is acceptable.

The v5 success claim should be framed as a Pareto result:

> SoloLLM v5 improves the local small-model tradeoff versus prior SoloLLM
> checkpoints and selected open baselines, measured across quality, size,
> speed, memory, and quantized local usability.
