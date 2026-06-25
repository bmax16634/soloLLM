# V4 Data Proxy: smollm_reasoning

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/_projects/soloLLM/data/v4_smollm_reasoning_300m_1024`
- Run: `/home/bmx/_projects/ariya/userdata/projects/soloLLM/outputs/sologpt_v4/data_proxy_smollm_reasoning_57m_300m`
- Accepted tokens: `300,000,591`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 3.8521 |
| Best val PPL | 47.09 |
| Final train loss | 4.0247 |
| Tok/s | 72178.3 |
| Hours | 1.15 |
| Peak GB | 12.3 |

Compare against the v3 pilot proxy baseline: best val loss `4.1530`, PPL `63.63`.
