# V4 Data Proxy: v4_1_blend

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/ariya-data/projects/soloLLM/data/v4_1_blend_300m_1024`
- Run: `/home/bmx/ariya-data/projects/soloLLM/outputs/sologpt_v4/data_proxy_v4_1_blend_57m_300m`
- Accepted tokens: `300,000,140`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 4.0682 |
| Best val PPL | 58.45 |
| Final train loss | 4.0757 |
| Tok/s | 72103.6 |
| Hours | 1.16 |
| Peak GB | 12.3 |

## Comparison

| Data mix | Best val loss | Best val PPL | Read |
|---|---:|---:|---|
| v3 pilot mix | 4.1530 | 63.63 | broad baseline |
| quality_web | 4.1926 | 66.20 | rejected |
| smollm_reasoning | 3.8521 | 47.09 | strongest proxy, but full-scale broad-PPL regression |
| v4.1 blend | 4.0682 | 58.45 | improves over v3 pilot, weaker than reasoning-heavy proxy |

## Takeaway

The v4.1 blend is a real proxy improvement over the v3 pilot data mix, but it
does not match the reasoning-heavy proxy's validation win. Because the
reasoning-heavy full run regressed on broad held-out PPL, this result should not
be scaled directly. The next step is to run the modern eval suite on this proxy
checkpoint and compare broad/general benchmarks before committing a full 150M
10B-token run.

## Modern Eval

The modern eval suite was run on the 57M proxy checkpoint.

| Model | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|
| v4 proxy `smollm_reasoning` 57M | 155.71 | 72.56 | 0.2271 | 0.3686 |
| v4.1 proxy `blend` 57M | 144.80 | 66.93 | 0.2408 | 0.3601 |

The blend improves broad external LM metrics but loses some multiple-choice
strength. This is useful, but not enough to justify scaling the exact mix.

Detailed reports:

- `docs/results/v4_proxy_v4_1_blend_57m/external_modern_baselines.md`
- `docs/results/v4_proxy_v4_1_blend_57m/multiple_choice_modern_baselines.md`
- `docs/results/v4_proxy_v4_1_blend_57m/summary.md`
