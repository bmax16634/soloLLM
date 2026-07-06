# V4 SmolLM-Reasoning 150M 10B Result

Date: July 6, 2026

This report records the full-scale v4 data-mix experiment. The goal was not to
replace the final v3 project retroactively; it was to test whether a
SmolLM-inspired reasoning-heavy mix could close the gap to modern small LMs
while keeping the same 151.9M-parameter SoloLLM architecture and RTX 3090
training setup.

## Bottom Line

V4 did not become the new best general SoloLLM model.

The reasoning-heavy data mix improved multiple-choice benchmark accuracy and
generation repetition metrics, but it hurt broad language-modeling quality
relative to v3 150M. The final project baseline should remain:

> v3 150M is the best general SoloLLM base model.

V4 is still useful as a research result:

> A reasoning-heavy educational/synthetic/math data mix can improve
> benchmark-style continuation scoring and reduce repetition, but it can
> trade away general held-out perplexity and broad base-LM quality.

## Run Summary

| Field | Value |
| --- | ---: |
| Model | `v4_smollm_reasoning_10b_150m` |
| Parameters | 151,868,928 |
| Context | 1024 |
| Architecture | v3-style RMSNorm/SwiGLU/RoPE decoder |
| Dataset | `v4_smollm_reasoning_10b_1024` |
| Total train tokens | 9,800,196,096 |
| Resume segment tokens | 5,800,173,568 |
| Avg throughput | 32.1k tok/s |
| Resume runtime | 50.23h |
| Peak VRAM | 13.8GB |
| Best train-val loss | 2.7790 |
| Best train-val PPL | 16.10 |

Training completed cleanly from the 4B checkpoint using unseen shards
`477:1168`. The eval initially failed because the project had moved and the
repo-local `data/tokenized_chunks` path was missing. After adding a local data
symlink to the external dataset store, the eval was rerun successfully.

## Data Mix

The full v4 run used the `smollm_reasoning` mixture from
`sologpt_v4/data_mix_smollm_reasoning_10b.yaml`, based on the proxy experiment
that had looked promising at 57M parameters:

| Source family | Role |
| --- | --- |
| SmolLM FineWeb-Edu dedup | high-quality educational web |
| Cosmopedia v2 | synthetic textbook/blog/story text |
| FineMath-4+ | math reasoning and worked explanations |
| DCLM baseline | curated web diversity |
| Wikipedia | reference prose |
| OpenWebText | legacy project distribution |

The 57M proxy result was strong: validation PPL improved from `63.63` on the
v3 pilot mix to `47.09` on `smollm_reasoning`. The full 150M result showed that
this proxy signal did not transfer cleanly to the final general-LM objective.

## GPT-2 Comparison

Lower perplexity is better. Higher accuracy is better.

| Benchmark | GPT-2 small | v4 150M | Result |
| --- | ---: | ---: | --- |
| Project held-out PPL | 25.32 | 27.38 | v4 loses |
| WikiText-2 PPL | 45.32 | 43.59 | v4 wins |
| LAMBADA PPL | 40.62 | 38.46 | v4 wins |
| LAMBADA last-token acc | 46.67% | 46.77% | v4 slight win |
| LAMBADA last-word acc | 32.60% | 32.91% | v4 slight win |
| MC avg acc norm | 41.05% | 43.52% | v4 wins |
| Repeated bigram fraction | 17.15% | 8.55% | v4 wins |
| Bad-loop count | 2 / 8 | 1 / 8 | v4 wins |

Read: v4 beats GPT-2 on several external and diagnostic metrics, but it loses
the main project held-out perplexity by about `8.1%` relative PPL.

## V3 Comparison

The key question is whether v4 replaces v3. It does not.

| Benchmark | v3 150M | v4 150M | Result |
| --- | ---: | ---: | --- |
| Project held-out PPL | 24.90 | 27.38 | v3 wins |
| WikiText-2 PPL | 41.18 | 43.59 | v3 wins |
| LAMBADA PPL | 35.35 | 38.46 | v3 wins |
| LAMBADA last-word acc | 33.07% | 32.91% | v3 slight win |
| MC avg acc norm | 42.71% | 43.52% | v4 wins |

V4 is about `10.0%` worse than v3 150M on project held-out perplexity. The only
clear full-suite win over v3 is multiple-choice average normalized accuracy.

## Multiple-Choice Detail

`Accuracy norm` is the primary read because it reduces continuation-length
bias.

| Benchmark | GPT-2 acc norm | v3 150M acc norm | v4 150M acc norm |
| --- | ---: | ---: | ---: |
| HellaSwag | 29.53% | 30.08% | 31.52% |
| PIQA | 63.60% | 62.70% | 63.60% |
| ARC-Easy | 40.35% | 44.56% | 47.02% |
| ARC-Challenge | 22.07% | 25.08% | 25.42% |
| WinoGrande | 49.72% | 51.14% | 50.04% |
| Average | 41.05% | 42.71% | 43.52% |

This is the strongest evidence that the reasoning-heavy mix did something real:
it improved HellaSwag, ARC-Easy, ARC-Challenge, and average normalized
multiple-choice scoring.

## Generation Metrics

The fixed prompt suite is small, so these metrics are diagnostic rather than
human-preference proof.

| Metric | GPT-2 small | v3 150M | v4 150M |
| --- | ---: | ---: | ---: |
| Corpus distinct-1 | 0.3869 | 0.4308 | 0.4738 |
| Corpus distinct-2 | 0.7836 | 0.8053 | 0.8831 |
| Mean unique token ratio | 0.5905 | 0.5759 | 0.6459 |
| Repeated bigram fraction | 0.1715 | 0.1723 | 0.0855 |
| Repeated trigram fraction | 0.1120 | 0.1006 | 0.0382 |
| Bad-loop count | 2 / 8 | 2 / 8 | 1 / 8 |

V4 clearly reduces repeated n-grams on the small fixed generation suite. This
does not override the poorer general LM perplexity, but it is a useful signal
for future data balancing.

## Interpretation

The v4 result points to a data-mix tradeoff:

- More reasoning/educational/synthetic/math data helped multiple-choice and
  repetition diagnostics.
- The same mix weakened broad held-out language modeling versus v3.
- The 57M proxy overestimated the full-run benefit because it measured local
  validation on the v4 data distribution, not full generalization to the fixed
  v3/GPT-2 suite.

This means the next experiment should not simply continue v4 longer. The issue
is not undertraining; the issue is objective/data balance.

## V4.1 Decision

If the project continues, the next run should be called `v4.1`, not `v5`.

V4.1 should be a bounded data-mix correction:

| Component | Suggested share | Purpose |
| --- | ---: | --- |
| v3-style broad web/reference data | 60-75% | recover general LM perplexity |
| SmolLM reasoning/educational/math data | 15-25% | keep MC/repetition gains |
| Wikipedia/reference/code-light high-signal data | 5-10% | improve factual/structured coverage |

Before another full 150M 10B run, run a cheap 57M proxy against:

1. v3 pilot mix,
2. v4 reasoning-heavy mix,
3. v4.1 blended mix.

Only scale v4.1 if the proxy improves both validation PPL and modern-suite
scores. Do not scale a mix that only improves multiple-choice while hurting
general perplexity.

## Artifacts

| Artifact | Path |
| --- | --- |
| Resume/train wrapper | `scripts/run_v4_smollm_reasoning_10b_150m_resume_4b_train_eval_notify.sh` |
| Training summary | `outputs/sologpt_v4/v4_smollm_reasoning_10b_plus_150m_1024/metrics_summary.json` |
| Final model | `outputs/sologpt_v4/v4_smollm_reasoning_10b_plus_150m_1024/final_model.pt` |
| Full eval output | `outputs/eval_suites/v4_smollm_reasoning_10b_150m_gpt2_full_suite/` |
| Eval retry log | `outputs/eval_suites/v4_smollm_reasoning_10b_150m_gpt2_full_suite/v3_eval_suite_full_retry.log` |
| Generation samples | `outputs/eval_suites/v4_smollm_reasoning_10b_150m_gpt2_full_suite/reports/generations_candidate_gpt2.md` |
| External benchmarks | `outputs/eval_suites/v4_smollm_reasoning_10b_150m_gpt2_full_suite/reports/external_full_candidate_gpt2.md` |
| Multiple choice | `outputs/eval_suites/v4_smollm_reasoning_10b_150m_gpt2_full_suite/reports/multiple_choice_full_candidate_gpt2.md` |
