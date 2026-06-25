# V4 Data Proxy: quality_web

57M v3-style proxy trained on a candidate v4 data mix.

- Dataset: `/home/bmx/_projects/soloLLM/data/v4_quality_web_300m_1024`
- Run: `/home/bmx/_projects/ariya/userdata/projects/soloLLM/outputs/sologpt_v4/data_proxy_quality_web_57m_300m`
- Accepted tokens: `300,000,431`

| Metric | Value |
|---|---:|
| Params | 57117696 |
| Tokens | 300011520 |
| Best val loss | 4.1926 |
| Best val PPL | 66.20 |
| Final train loss | 4.2038 |
| Tok/s | 72131.7 |
| Hours | 1.16 |
| Peak GB | 12.3 |

Compare against the v3 pilot proxy baseline: best val loss `4.1530`, PPL `63.63`.
