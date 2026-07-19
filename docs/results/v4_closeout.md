# V4 Research Closeout

Date: July 19, 2026

## Bottom Line

V4 is closed.

The v4 research phase tested whether post-v3 data-mix and training-recipe
changes could improve SoloLLM beyond the public v3 release. It produced one
clear best checkpoint and one smaller-model ablation:

- `v4.2-dropout0-v4.1-blend-150m` is the best quantitative SoloLLM checkpoint.
  It beats GPT-2 small across the fixed project suite, but it is larger than
  GPT-2 small.
- `v4.2-dropout0-v4.1-blend-123m` is slightly smaller than GPT-2 small. It
  wins all five multiple-choice checks and most external checks, but it does
  not beat GPT-2 across every metric.

The strict smaller-than-GPT-2 across-board claim remains unproven. Further v4
work should not chase small decimal-point wins on the same GPT-2 suite. The
next phase should use a broader small-model research frame with stronger
baselines, locked evals, inference metrics, and local-usefulness targets.

## Completed Runs

| Run | Params | Train tokens | Status |
| --- | ---: | ---: | --- |
| `v4-smollm-reasoning-150m` | 151.87M | 9.80B | Useful failed ablation: MC/repetition improved, broad PPL regressed |
| `v4.2-dropout0-v4.1-blend-150m` | 151.87M | 9.80B | Best quantitative SoloLLM checkpoint; beats GPT-2 small across fixed suite |
| `v4.2-dropout0-v4.1-blend-123m` | 123.55M | 9.80B | Smaller-than-GPT-2 ablation; wins MC suite and most external checks, but not all metrics |

## Headline Comparison

| Model | Params | Train tokens | Held-out PPL | WikiText-2 PPL | LAMBADA PPL | LAMBADA tok acc | LAMBADA word acc | MC avg norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 124.44M | public | 25.32 | 45.32 | 40.62 | 46.67% | 32.60% | 41.05% |
| SoloLLM v3 150M | 151.87M | 10.00B | 24.90 | 41.18 | 35.35 | 47.40% | 33.07% | 42.71% |
| SoloLLM v4 reasoning 150M | 151.87M | 9.80B | 27.38 | 43.59 | 38.46 | 47.09% | 32.91% | 43.52% |
| SoloLLM v4.2 blend 150M | 151.87M | 9.80B | 24.30 | 41.75 | 34.06 | 47.27% | 33.05% | 43.61% |
| SoloLLM v4.2 blend 123M | 123.55M | 9.80B | 25.56 | 43.75 | 35.45 | 46.30% | 31.69% | 43.13% |

Read:

- The 150M v4.2 run is the strongest overall quantitative model.
- The 123M v4.2 run is a useful size-controlled ablation, not a clean win.
- The main 123M misses are small but real: project held-out PPL, LAMBADA
  last-token accuracy, and LAMBADA greedy exact-word accuracy.

## V4.2 123M Run Summary

| Item | Value |
| --- | --- |
| Model | `v4_2_dropout0_v4_1_blend_123m_1024` |
| Parameters | `123,551,232` |
| GPT-2 small parameters | `124,439,808` |
| Context | `1024` |
| Dropout | `0.0` |
| Train tokens | `9,800,572,928` |
| Final train loss | `3.0258` |
| Best internal validation loss | `2.9983` |
| Best internal validation PPL | `20.05` |
| Average train speed | `40.1k` tok/s |
| Peak VRAM | `11.6GB` |
| Training time | `67.86h` |
| Training finished | `2026-07-18 16:38 UTC` |
| Eval finished | `2026-07-18 20:26 UTC` |
| Final model | `outputs/sologpt_v4/v4_2_dropout0_v4_1_blend_10b_123m_1024/final_model.pt` |
| Eval output | `outputs/eval_suites/v4_2_dropout0_v4_1_blend_10b_123m_gpt2_full_suite/` |

## V4.2 123M Multiple Choice

| Benchmark | GPT-2 acc norm | V4.2 123M acc norm | Delta |
| --- | ---: | ---: | ---: |
| HellaSwag | 29.53% | 31.92% | +2.39 |
| PIQA | 63.60% | 65.00% | +1.40 |
| ARC-Easy | 40.35% | 44.21% | +3.86 |
| ARC-Challenge | 22.07% | 24.08% | +2.01 |
| WinoGrande | 49.72% | 50.43% | +0.71 |
| Average | 41.05% | 43.13% | +2.08 |

The 123M v4.2 model beats GPT-2 small on every normalized multiple-choice
benchmark in the fixed suite.

## V4.2 123M Generation Metrics

Metrics use the same eight fixed prompts as earlier diagnostics.

| Model | Distinct-1 | Distinct-2 | Repeated bigrams | Repeated trigrams | Bad loops |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | 0.3869 | 0.7836 | 0.1715 | 0.1120 | 2 |
| SoloLLM v4.2 123M | 0.4535 | 0.8453 | 0.1347 | 0.0618 | 2 |

These support better diversity and lower repetition on the fixed prompts, but
they are not a substitute for broader qualitative review.

## Final Interpretation

V4 answered the GPT-2 comparison question well enough to stop this phase:

- A from-scratch SoloLLM model can beat GPT-2 small across the fixed suite
  when allowed a modest parameter increase to 151.9M.
- A slightly smaller-than-GPT-2 SoloLLM model can beat GPT-2 on all fixed
  multiple-choice checks and most external language-modeling checks, but not
  across every metric.
- Repeatedly tuning against the same GPT-2 suite now risks benchmark-chasing.

The recommended next phase is a separate v5 small-model research program:
compare against modern small baselines, freeze a broader eval suite, measure
local inference and quantized behavior, and optimize for useful quality on
single-consumer-GPU hardware rather than only GPT-2 parity.
