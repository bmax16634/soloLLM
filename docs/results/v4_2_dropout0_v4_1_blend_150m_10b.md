# V4.2 Dropout-0 V4.1-Blend 150M 10B Result

Date: July 15, 2026

## Bottom Line

V4.2 is the strongest quantitative SoloLLM checkpoint so far. It beats GPT-2
small on the fixed project suite: project held-out perplexity, WikiText-2,
LAMBADA perplexity, LAMBADA continuation accuracy, all five length-normalized
multiple-choice checks, and fixed-prompt repetition/diversity support metrics.

This is not a smaller-than-GPT-2 result. V4.2 has `151,868,928` parameters,
while GPT-2 small has `124,439,808`. The correct claim is that the dropout-0
broad-blend recipe produced the best quantitative SoloLLM model so far, not that
it proved a smaller model can beat GPT-2 across the board.

## Run Summary

| Item | Value |
| --- | --- |
| Model | `v4_2_dropout0_v4_1_blend_150m_1024` |
| Parameters | `151,868,928` |
| Context | `1024` |
| Dropout | `0.0` |
| Tokenizer | GPT-2 BPE |
| Dataset | fresh v4.1-style broad blend, 10B target |
| Train tokens | `9,800,572,928` |
| Final train loss | `3.0599` |
| Best internal validation loss | `2.9500` |
| Best internal validation PPL | `19.11` |
| Final model | `outputs/sologpt_v4/v4_2_dropout0_v4_1_blend_10b_150m_1024/final_model.pt` |
| Eval output | `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_150m_gpt2_full_suite/` |

Training was interrupted by a power outage after the 8B checkpoint. The run
resumed from `ckpt_8000M.pt` and completed the remaining unseen training shards.
Resume-segment throughput averaged about `32.6k` tokens/sec over `15.34h`.

## Dataset

The 10B-token dataset was built from the v4.1-style broad blend selected by the
proxy experiments:

| Source | Weight |
| --- | ---: |
| FineWeb-Edu sample-10BT | 30% |
| DCLM baseline | 20% |
| FineWeb sample-10BT | 15% |
| Cosmopedia v2 | 10% |
| Wikipedia English | 8% |
| OpenWebText | 7% |
| FineMath 4+ | 7% |
| SmolLM FineWeb-Edu dedup | 3% |

Split metadata:

| Split | Shards | Tokens |
| --- | ---: | ---: |
| Train | 1169 | `9,800,571,904` |
| Validation | 12 | `99,416,064` |
| Test | 12 | `100,012,032` |

## Headline Comparison

| Model | Params | Train tokens | Held-out PPL | WikiText-2 PPL | LAMBADA PPL | LAMBADA word acc | MC avg norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 124.44M | public | 25.32 | 45.32 | 40.62 | 32.60% | 41.05% |
| SoloLLM v3 123M | 123.55M | 9.80B | 25.64 | 41.87 | 36.28 | 32.80% | 42.46% |
| SoloLLM v3 150M | 151.87M | 10.00B | 24.90 | 41.18 | 35.35 | 33.07% | 42.71% |
| SoloLLM v4 reasoning 150M | 151.87M | 9.80B | 27.38 | 43.59 | 38.46 | 32.91% | 43.52% |
| SoloLLM v4.2 blend 150M | 151.87M | 9.80B | 24.30 | 41.75 | 34.06 | 33.05% | 43.61% |

Read: the first v4 reasoning-heavy full run improved multiple-choice and
repetition metrics but lost too much broad language-modeling quality. V4.2
keeps the multiple-choice gain while recovering and improving the broad
perplexity profile.

## Project Held-Out Perplexity

Full held-out eval used reserved project shards `58:60`, `331,353,862` tokens.

| Model | Loss | PPL | Params |
| --- | ---: | ---: | ---: |
| GPT-2 small | 3.2314 | 25.3150 | 124.44M |
| SoloLLM v4.2 blend 150M | 3.1906 | 24.3035 | 151.87M |

Per-shard v4.2 perplexities were stable: `24.2344`, `24.3958`, and `24.1966`.

## External Base-LM Checks

| Benchmark | Model | BPB | PPL | Last-token acc | Last-word exact |
| --- | --- | ---: | ---: | ---: | ---: |
| WikiText-2 | GPT-2 small | 1.2216 | 45.3195 | n/a | n/a |
| WikiText-2 | SoloLLM v4.2 | 1.1953 | 41.7512 | n/a | n/a |
| LAMBADA | GPT-2 small | 1.3764 | 40.6243 | 46.67% | 32.60% |
| LAMBADA | SoloLLM v4.2 | 1.3109 | 34.0612 | 47.27% | 33.05% |

These are lightweight base-LM checks, not a replacement for a full
lm-eval-harness paper benchmark.

## Multiple Choice

Choices are scored by conditional log-likelihood. `Accuracy norm` uses average
log-probability per continuation token to reduce length bias.

| Benchmark | GPT-2 acc norm | V4.2 acc norm | Delta |
| --- | ---: | ---: | ---: |
| HellaSwag | 29.53% | 32.99% | +3.46 |
| PIQA | 63.60% | 65.50% | +1.90 |
| ARC-Easy | 40.35% | 45.09% | +4.74 |
| ARC-Challenge | 22.07% | 23.08% | +1.01 |
| WinoGrande | 49.72% | 51.38% | +1.66 |
| Average | 41.05% | 43.61% | +2.56 |

## Fixed-Prompt Generation Metrics

Metrics use the same eight fixed prompts as earlier v2/v3/v4 diagnostics.

| Model | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 0.3869 | 0.7836 | 0.1715 | 0.1120 | 2 |
| SoloLLM v4.2 | 0.4000 | 0.8109 | 0.1534 | 0.0836 | 2 |

These metrics support the quantitative win, but they do not prove qualitative
parity. Manual samples still show that factuality, logic, and code consistency
are not at modern assistant quality.

## Conclusion

The v4.2 result changes the best-current-model framing:

- V3 remains the public Hugging Face release and the completed project closeout.
- V4 reasoning-heavy data alone was a useful failed ablation: better MC and
  lower repetition, worse general perplexity.
- V4.2 dropout-0 on a broader v4.1-style blend is the best quantitative
  SoloLLM checkpoint and beats GPT-2 small across this fixed suite.
- The unresolved research goal is the smaller-model claim: the 123M v3 model
  comes close and wins many external checks, but the across-board win still
  requires a model at or below GPT-2-small scale.
