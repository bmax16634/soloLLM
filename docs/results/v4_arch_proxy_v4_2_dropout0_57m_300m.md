# V4 Data Proxy: v4_2_dropout0

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/ariya-data/projects/soloLLM/data/v4_1b_blend_300m_1024`
- Run: `/home/bmx/ariya-data/projects/soloLLM/outputs/sologpt_v4/arch_proxy_v4_2_dropout0_57m_300m`
- Accepted tokens: `300,000,724`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 3.9373 |
| Best val PPL | 51.28 |
| Final train loss | 4.0330 |
| Tok/s | 73043.5 |
| Hours | 1.14 |
| Peak GB | 12.1 |

## Comparison

| Proxy | Val PPL | WikiText PPL | LAMBADA PPL | MC avg norm |
|---|---:|---:|---:|---:|
| v4.1b dropout0.1 | 54.46 | 149.37 | 68.55 | 0.3613 |
| v4.2 dropout0.0 | 51.28 | 149.84 | 65.44 | 0.3678 |

## Decision

Dropout `0.0` improves the 57M proxy more than the data-only v4.1b adjustment
did. It should be carried forward into the next proxy, but it does not yet
justify a full 150M/10B train by itself because WikiText is not improved.
