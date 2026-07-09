# V4 Data Proxy: v4_2_dropout0_v4_1_blend

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/ariya-data/projects/soloLLM/data/v4_1_blend_300m_1024`
- Run: `/home/bmx/ariya-data/projects/soloLLM/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_v4_1_blend_57m_300m`
- Accepted tokens: `300,000,140`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 4.0069 |
| Best val PPL | 54.98 |
| Final train loss | 3.9285 |
| Tok/s | 73012.6 |
| Hours | 1.14 |
| Peak GB | 12.1 |

## Comparison

| Proxy | Val PPL | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|---:|
| v4.1 blend dropout0.1 | 58.45 | 144.80 | 66.93 | 0.2408 | 0.3601 |
| v4.1b blend dropout0.1 | 54.46 | 149.37 | 68.55 | 0.2381 | 0.3613 |
| v4.1b blend dropout0.0 | 51.28 | 149.84 | 65.44 | 0.2366 | 0.3678 |
| v4.1 blend dropout0.0 | 54.98 | 141.55 | 64.54 | 0.2420 | 0.3705 |

## Decision

Scale candidate. This is the first proxy that improves the broad v4.1 blend
without giving back MC. The next full run should use a fresh 10B-token v4.1
blend dataset and the dropout-0.0 recipe.
