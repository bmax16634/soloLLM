# SoloLLM Modern Gap Plan

SoloLLM v3 is a complete GPT-2-class training milestone: the final 150M model beats GPT-2 small on the project eval suite and on the expanded modern-baseline run. The next serious research direction is not "beat GPT-2 again." The useful target is closing the gap to modern small base models, especially SmolLM2-135M.

## Research Question

Which changes most effectively close the gap between SoloLLM v3 and modern small LMs under a single RTX 3090 training budget: architecture shape, data quality/mix, context length, or training recipe?

The output should be both:

- a stronger released base model
- a controlled technical report showing what actually moved the metrics

## Current Baseline

Modern-baseline run:

- Script: `scripts/run_modern_baselines_eval_notify.sh`
- Reports: `docs/results/modern_baselines_v3_150m/`
- Candidate: SoloLLM v3 150M, trained from scratch on the 10B-token v3 dataset
- Baselines: GPT-2 small, DistilGPT2, Pythia 70M/160M, SmolLM 135M, SmolLM2 135M, SmolLM2 360M
- Tokenizer policy: each model uses its native tokenizer
- Cross-tokenizer LM metric: bits-per-byte, not raw token perplexity

### Key Result

SoloLLM v3 150M beats GPT-2 small but trails SmolLM/SmolLM2.

| Model | WikiText BPB | LAMBADA BPB | LAMBADA word acc | MC avg norm |
|---|---:|---:|---:|---:|
| SmolLM2 360M | 1.025 | 1.200 | 53.25% | 56.77% |
| SmolLM2 135M | 1.107 | 1.262 | 42.67% | 49.78% |
| SmolLM 135M | 1.185 | 1.349 | 37.71% | 48.47% |
| SoloLLM v3 150M | 1.191 | 1.325 | 33.07% | 42.71% |
| GPT-2 small | 1.222 | 1.376 | 32.60% | 41.05% |

Against SmolLM2-135M, SoloLLM v3 is:

- 7.6% worse on WikiText BPB
- 5.0% worse on LAMBADA BPB
- 9.6 points lower on LAMBADA last-word accuracy
- 7.1 points lower on average normalized multiple-choice accuracy
- 11.0 points lower on HellaSwag normalized accuracy
- 12.1 points lower on ARC-Easy normalized accuracy

## Leading Hypotheses

### 1. Architecture Shape Matters

SoloLLM v3 150M:

- 16 layers
- 768 hidden size
- 12 attention heads
- 1024 context
- GPT-2 tokenizer
- RMSNorm, SwiGLU, RoPE

SmolLM2-135M:

- 30 layers
- 576 hidden size
- 9 attention heads
- 3 KV heads
- 8192 context
- 49k tokenizer
- LLaMA-style architecture family

Hypothesis: for this parameter range, a deeper/narrower grouped-query-attention model may outperform the current wider/shallower SoloLLM shape, especially on reasoning-style multiple-choice tasks.

### 2. Data Quality and Mix Matter More Than More Tokens Alone

The biggest gaps are HellaSwag, ARC, and LAMBADA. That suggests the current data mix is strong enough for GPT-2-era LM loss, but weak against modern small models on commonsense, educational, and continuation-style tasks.

Hypothesis: a cleaner education/reference/web mix will move benchmark accuracy more than simply repeating the current v3 data longer.

### 3. Context Length May Be a Secondary Bottleneck

SoloLLM v3 was evaluated at 1024 context. SmolLM2-135M supports much longer context. The immediate target should be 2048 context because it is more realistic on an RTX 3090; 4096+ can be tested only after the architecture/data question is clearer.

## Experiment Plan

### Phase 0: Lock Evaluation

Keep the modern-baseline suite fixed:

- WikiText BPB
- LAMBADA BPB
- LAMBADA last-token accuracy
- LAMBADA last-word exact accuracy
- HellaSwag
- PIQA
- ARC-Easy
- ARC-Challenge
- WinoGrande
- fixed prompt generations and generation metrics

Primary comparison target:

- SmolLM2-135M

Secondary baselines:

- GPT-2 small
- SmolLM-135M
- Pythia-160M
- SmolLM2-360M as an upper-bound reference, not a fair same-size target

### Phase 1: Architecture Proxy

Goal: test whether SmolLM2-style shape helps before spending days on a full train.

Run two proxy models on the same data and token budget:

| Proxy | Shape | Purpose |
|---|---|---|
| v3-style proxy | wider/shallower | current architecture control |
| SmolLM2-style proxy | deeper/narrower, GQA if supported | architecture hypothesis |

Suggested budget:

- 40M-60M parameters
- 300M-1B tokens
- same train/val split
- eval every major checkpoint

Decision rule:

- If SmolLM2-style proxy wins consistently on BPB and multiple-choice, use that shape for the next full model.
- If not, prioritize data curation before architecture work.

### Phase 2: Data Proxy

Use the better proxy architecture from Phase 1 and compare data recipes:

- current v3 balanced data
- cleaner/high-quality web
- education-heavy mix
- wiki/reference-heavy mix
- code/math-light mix
- aggressive filtering/dedup variant

Decision rule:

- Choose the recipe that improves both BPB and multiple-choice accuracy.
- Do not optimize only WikiText or only LAMBADA.

### Phase 3: Final SoloLLM-Modern Train

Train the best recipe at 123M-150M scale first.

Success criteria:

- beat GPT-2 small, already achieved by v3
- close most of the SmolLM2-135M gap
- improve HellaSwag and ARC substantially over v3
- release model, model card, eval report, and demo

Stretch target:

- match or beat SmolLM-135M

Hard target:

- match SmolLM2-135M across most metrics

## What Not To Do Yet

- Do not jump straight to 300M-500M parameters.
- Do not change architecture, tokenizer, data, and optimizer all at once.
- Do not treat a better demo as proof of a better base model.
- Do not claim a research contribution from one final model alone.

## Immediate Next Step

Implement the Phase 1 architecture proxy:

1. Add a SmolLM2-style small config.
2. Add grouped-query attention support if the current model does not support it.
3. Run a tiny forward/training smoke test.
4. Launch a controlled proxy run against the v3-style proxy.
