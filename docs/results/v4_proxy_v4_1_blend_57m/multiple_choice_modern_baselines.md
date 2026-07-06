# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under each base LM's native tokenizer. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `1024`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | solollm-v4-1-blend-proxy-57m | 10042 | 0.2577 | 0.2476 | 125.9900 |
| piqa | solollm-v4-1-blend-proxy-57m | 1000 | 0.5610 | 0.5610 | 48.1460 |
| arc_easy | solollm-v4-1-blend-proxy-57m | 570 | 0.3281 | 0.2860 | 17.6982 |
| arc_challenge | solollm-v4-1-blend-proxy-57m | 299 | 0.1472 | 0.2040 | 23.0669 |
| winogrande | solollm-v4-1-blend-proxy-57m | 1267 | 0.4988 | 0.5020 | 14.4167 |
