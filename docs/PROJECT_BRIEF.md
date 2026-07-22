# SoloLLM Project Brief

## Problem

SoloLLM is a from-scratch GPT-style language-model project built to show the full engineering loop behind a small decoder-only LM: data preparation, model design, training infrastructure, evaluation, and serving.

The goal is not to ship a production assistant. The goal is to make the work reproducible and technically legible enough that another engineer can inspect the system, run smoke tests, and understand what improved from v1 to v2 to v3.

## V1 Baseline

V1 proved the core path worked:

- custom PyTorch decoder-only model,
- OpenWebText tokenization and flattened shards,
- pretraining script,
- Dolly fine-tuning path,
- Streamlit generation demo,
- published Hugging Face artifact.

The weakness of v1 is that it was a rough prototype. The architecture, training loop, eval path, and run artifacts were not yet strong enough for a polished ML systems portfolio case study.

## V2 Design

V2 is the serious rebuild. It now has two intended variants:

| Variant | Purpose | Target |
| --- | --- | --- |
| `v2-gpt2-parity` | Control model | 123,616,512 params, GPT-2-small scale |
| `v2-modern-small` | Main experiment | 91,654,400 params, smaller than GPT-2-small |

The parity model is intentionally small and defensible:

- 12 transformer layers,
- 768 embedding width,
- 12 attention heads,
- 512-token context,
- 123,616,512 parameters with tied embeddings,
- RoPE attention,
- pre-norm blocks,
- optional tied input/output embeddings,
- explicit max-context validation.

This roughly matches GPT-2-small scale, which makes the final GPT-2 comparison easier to explain than simply increasing parameter count.

The modern-small target is the original consumer-hardware experiment:

- train on a single RTX 3090,
- use fewer parameters than GPT-2-small,
- add RMSNorm,
- add SwiGLU,
- keep RoPE and tied embeddings,
- test whether a smaller modern model can approach or beat GPT-2-small on held-out perplexity.

Phase 1 sanity results favored `v2-modern-small`: it was smaller, faster, and reached lower capped validation perplexity than the GPT-2-parity control after 50M tokens. The result is documented in `docs/results/phase1_50m_sanity.md`.

## Training System

The v2 trainer is designed around reproducibility and recovery:

- CLI-driven configuration,
- fixed train/validation/test shard ranges,
- CPU-safe dry run,
- JSONL metrics,
- resolved config copy per run,
- checkpoint metadata,
- `checkpoints/latest.pt`,
- resume support,
- final model artifact,
- metrics summary JSON.

The intended split is:

| Split | Shards |
| --- | --- |
| Train | `0:54` |
| Validation | `55:57` |
| Test | `58:60` |

## Evaluation

Final portfolio evaluation compares:

| Model | Params | Context | Training Tokens | Eval Split | Loss | Perplexity | Notes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| SoloGPT v1 | 203.75M | 512 | 9.03B declared target, actual unverified | `58:60` | 3.41697 | 30.48 | Historical baseline, HF checkpoint |
| v2-modern-small | 91.65M | 512 | 3.00B confirmed | `58:60` | 3.26816 | 26.26 | Smaller modern experiment |
| v2-modern-small stretch | 91.65M | 512 | 5.60B checkpoint metadata | `58:60` | 3.24083 | 25.56 | Best official v2 checkpoint |
| GPT-2 small | 124.44M | 1024 | Public pretrained | `58:60` | 3.23140 | 25.32 | Reference baseline |

The best v2 checkpoint beats v1 clearly with about 55% fewer parameters. It improves over the 3B checkpoint, gets within about 0.95% perplexity of GPT-2 small, and still does not beat GPT-2. The final writeup should state all parts directly.

The 5.60B stretch checkpoint full eval was run on June 8, 2026:

| Model | Params | Checkpoint / Training Tokens | Eval Tokens | Loss | Perplexity | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| v2-modern-small stretch | 91.65M | 5.60B checkpoint metadata | 331,353,862 | 3.24083 | 25.56 | `checkpoints/latest.pt` |
| GPT-2 small | 124.44M | Public pretrained | 331,353,862 | 3.23140 | 25.32 | Reference baseline |

On the full held-out split, the 5.60B v2 checkpoint is about 16.1% lower perplexity than v1, about 2.7% lower perplexity than the 3B checkpoint, and about 0.95% higher perplexity than GPT-2.

Additional 5.60B comparison checks were added after the full held-out result:

| Check | v2-modern-small 5.60B | GPT-2 small | Interpretation |
| --- | ---: | ---: | --- |
| Fixed prompts corpus distinct-2 | 0.7512 | 0.7836 | similar diversity, GPT-2 slightly higher |
| Fixed prompts repeated bigram fraction | 0.2015 | 0.1715 | GPT-2 slightly less repetitive |
| WikiText-2 PPL, full test | 84.81 | 49.86 | GPT-2 much stronger out-of-domain |
| LAMBADA PPL, full test | 63.03 | 42.26 | GPT-2 stronger |
| LAMBADA last-token accuracy | 44.30% | 46.67% | close, GPT-2 ahead |
| LAMBADA last-word greedy exact | 29.09% | 32.60% | close, GPT-2 ahead |
| Multiple-choice length-normalized checks | trails all 5 | leads all 5 | GPT-2 more robust |

Generation samples, metrics, and the full diagnostic are saved in `docs/results/phase4_generations_5p6b_v2_gpt2.md`, `docs/results/phase4_generation_metrics_5p6b_v2_gpt2.md`, `docs/results/external_benchmarks_5p6b_v2_gpt2_full.md`, `docs/results/multiple_choice_5p6b_v2_gpt2_full.md`, and `docs/results/v2_gpt2_full_diagnostic.md`. They support a sharper conclusion: v2 is close to GPT-2 on the project held-out split, but GPT-2 remains more robust across broader external checks.

The frozen v2 closeout suite completed on June 9, 2026 at `03:24 UTC` and is stored at `outputs/eval_suites/v2_5p6b_gpt2_full_suite/`. That run is the baseline v3 should beat before making any claim stronger than "v2 approached GPT-2 on the project distribution."

## V3 Closeout

V3 used the v2 diagnostic gaps to target a broader result rather than simply continuing v2 longer. The main changes were:

- a curated 10B-token, 1024-context dataset,
- FineWeb-Edu, DCLM, FineWeb, Wikipedia, and OpenWebText source mix,
- GPT-2-scale and modestly larger modern model configs,
- a fixed full eval suite against GPT-2 small.

Final v3 checkpoints:

| Model | Params | Train tokens | Role |
| --- | ---: | ---: | --- |
| `v3-gpt2-scale-1024` | 123,551,232 | 9.80B | smaller-than-GPT-2 ablation |
| `v3-plus-150m-1024` | 151,868,928 | 10.00B | public final v3 model |

Final comparison:

| Benchmark | GPT-2 small | v3 123M | v3 150M |
| --- | ---: | ---: | ---: |
| Project held-out PPL | 25.32 | 25.64 | 24.90 |
| WikiText-2 PPL | 45.32 | 41.87 | 41.18 |
| LAMBADA PPL | 40.62 | 36.28 | 35.35 |
| LAMBADA last-word accuracy | 32.60% | 32.80% | 33.07% |
| Multiple-choice avg acc norm | 41.05% | 42.46% | 42.71% |

The public final v3 model is the 150M checkpoint, which beats GPT-2 small overall on the fixed suite. The smaller 123M checkpoint is about 0.7% smaller than GPT-2 and beats GPT-2 on most external checks, but it does not win project held-out perplexity or every generation metric. The strict "smaller than GPT-2 and better across the board" hypothesis is therefore not fully proven.

The final v3 report is `docs/results/v3_final_gpt2_comparison.md`. The retained
local datasets, checkpoints, metrics, and eval outputs are indexed in
`docs/V3_ARTIFACT_MANIFEST.md`.

Published artifacts:

| Artifact | Link |
| --- | --- |
| SoloLLM Hugging Face collection | <https://huggingface.co/collections/bmax16634/solollm> |
| Final v3 150M base model | <https://huggingface.co/bmax16634/sologpt-v3-150m-base> |
| v3 123M smaller-model ablation | <https://huggingface.co/bmax16634/sologpt-v3-123m-base> |
| Public v3 completion demo | <https://huggingface.co/spaces/bmax16634/sologpt-v3-150m-demo> |

## Post-V3 Research: V4

After v3 closeout, the project branch `v4-research` tested whether the remaining
gap to modern small LMs was primarily architecture shape or data mix.

The architecture proxy did not support switching to a deeper/narrower
SmolLM2-style shape by itself. A later 57M data proxy showed a large validation
gain for a SmolLM-inspired reasoning-heavy mix, so that mix was scaled to a full
151.9M-parameter 10B-token run.

The first full v4 result was mixed:

| Benchmark | GPT-2 small | v3 150M | v4 150M |
| --- | ---: | ---: | ---: |
| Project held-out PPL | 25.32 | 24.90 | 27.38 |
| WikiText-2 PPL | 45.32 | 41.18 | 43.59 |
| LAMBADA PPL | 40.62 | 35.35 | 38.46 |
| LAMBADA last-word accuracy | 32.60% | 33.07% | 32.91% |
| Multiple-choice avg acc norm | 41.05% | 42.71% | 43.52% |
| Repeated bigram fraction | 17.15% | 17.23% | 8.55% |

V4 improved multiple-choice scoring and fixed-prompt repetition metrics, but it
hurt broad general-LM perplexity versus v3.

The first v4 research finding was:

> A reasoning-heavy educational/synthetic/math mix can improve benchmark-style
> continuation scoring and reduce repetition, but it can trade away broad
> language-modeling quality. The next experiment should be a blended v4.1 data
> mix, not longer training of the completed v4 checkpoint.

The follow-up v4.2 proxy search kept the same 150M v3-style architecture but
changed the training recipe and data blend:

- dropout `0.1` to `0.0`,
- broader v4.1-style data blend instead of the reasoning-heavy mix,
- fresh 10B-token dataset,
- fresh 151.9M-parameter run from scratch.

The full v4.2 run became the best quantitative SoloLLM checkpoint:

| Benchmark | GPT-2 small | v3 150M | v4 150M | v4.2 150M | v4.2 123M |
| --- | ---: | ---: | ---: | ---: | ---: |
| Project held-out PPL | 25.32 | 24.90 | 27.38 | 24.30 | 25.56 |
| WikiText-2 PPL | 45.32 | 41.18 | 43.59 | 41.75 | 43.75 |
| LAMBADA PPL | 40.62 | 35.35 | 38.46 | 34.06 | 35.45 |
| LAMBADA last-word accuracy | 32.60% | 33.07% | 32.91% | 33.05% | 31.69% |
| Multiple-choice avg acc norm | 41.05% | 42.71% | 43.52% | 43.61% | 43.13% |
| Repeated bigram fraction | 17.15% | 17.23% | 8.55% | 15.34% | 13.47% |

The final v4 research finding is:

> Reasoning-heavy data alone improved multiple-choice and repetition metrics but
> damaged broad PPL. A broader v4.1-style blend plus dropout `0.0` recovered
> broad language-modeling quality and produced the strongest quantitative
> checkpoint at 150M parameters. The 123M ablation is slightly smaller than
> GPT-2 and wins the fixed multiple-choice suite, but it does not beat GPT-2
> across every metric.

Reports: `docs/results/v4_smollm_reasoning_150m_10b.md` and
`docs/results/v4_2_dropout0_v4_1_blend_150m_10b.md`. Final closeout:
`docs/results/v4_closeout.md`.

## Current Verification

Run:

```bash
python -m pytest
```

The smoke suite covers:

- v1 forward shape,
- v2 forward shape,
- v2 max-context rejection,
- v2 tied embeddings,
- v2 parameter-count summary,
- v2 dry-run training,
- checkpoint resume,
- eval CLI JSON output.

## Portfolio Framing

Resume version:

> Built from-scratch GPT-style language models in PyTorch on a single RTX 3090, curated a 10B-token training dataset, and trained GPT-2-class base LMs with reproducible checkpoints, full evals, and documented comparisons against GPT-2 small.

Case-study version:

> V1 proved the pipeline could work. V2 turned the project into an ML systems case study. V3 turned it into a full empirical comparison: data curation, scaling decisions, fixed GPT-2 evals, a 150M public model that wins overall, and a smaller 123M ablation that shows exactly where the strict smaller-model claim still fails. V4 research added controlled data-mix and training-recipe experiments, ending with a v4.2 150M checkpoint that beats GPT-2 small across the fixed quantitative suite and a 123M checkpoint that is smaller than GPT-2 and wins the MC suite but not every metric.

## Remaining Work

The v3 closeout, result docs, Hugging Face model repos, and public demo are
complete. V4 is closed locally. Publishing v4.2 checkpoints is optional. The
next research frame is documented in `docs/research/v5_small_model_research_plan.md`.
V5 should begin with a broader baseline/eval board, not a new long training run.
