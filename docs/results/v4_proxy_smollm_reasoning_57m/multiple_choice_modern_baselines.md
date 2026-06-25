# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under each base LM's native tokenizer. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `1024`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | solollm-v4-proxy-smollm-reasoning-57m | 10042 | 0.2597 | 0.2514 | 125.9900 |
| piqa | solollm-v4-proxy-smollm-reasoning-57m | 1000 | 0.5620 | 0.5680 | 48.1460 |
| arc_easy | solollm-v4-proxy-smollm-reasoning-57m | 570 | 0.3667 | 0.3123 | 17.6982 |
| arc_challenge | solollm-v4-proxy-smollm-reasoning-57m | 299 | 0.1639 | 0.1973 | 23.0669 |
| winogrande | solollm-v4-proxy-smollm-reasoning-57m | 1267 | 0.5036 | 0.5138 | 14.4167 |
