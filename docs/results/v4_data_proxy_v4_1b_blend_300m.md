# V4 Data Proxy: v4_1b_blend

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/ariya-data/projects/soloLLM/data/v4_1b_blend_300m_1024`
- Run: `/home/bmx/ariya-data/projects/soloLLM/outputs/sologpt_v4/data_proxy_v4_1b_blend_57m_300m`
- Accepted tokens: `300,000,724`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 3.9975 |
| Best val PPL | 54.46 |
| Final train loss | 4.1755 |
| Tok/s | 72146.8 |
| Hours | 1.16 |
| Peak GB | 12.3 |

## Comparison

| Data mix | Best val loss | Best val PPL | Read |
|---|---:|---:|---|
| v3 pilot mix | 4.1530 | 63.63 | broad baseline |
| smollm_reasoning | 3.8521 | 47.09 | strongest proxy validation, but full-scale broad-PPL regression |
| v4.1 blend | 4.0682 | 58.45 | best broad external proxy so far |
| v4.1b blend | 3.9975 | 54.46 | better internal validation than v4.1 |

## Modern Eval

| Proxy | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|
| smollm_reasoning | 155.71 | 72.56 | 0.2271 | 0.3686 |
| v4.1 blend | 144.80 | 66.93 | 0.2408 | 0.3601 |
| v4.1b blend | 149.37 | 68.55 | 0.2381 | 0.3613 |

## Decision

Do not scale the exact v4.1b mix to a full 150M/10B run. It improves internal
proxy validation versus v4.1 and barely improves MC average, but gives back some
of v4.1's broad external gains. The data-mix tradeoff is now clear enough that
another full 150M data-only run is not justified without a new variable.
