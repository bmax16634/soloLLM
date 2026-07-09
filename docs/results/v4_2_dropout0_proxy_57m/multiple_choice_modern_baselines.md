# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under each base LM's native tokenizer. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `1024`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | solollm-v4-2-dropout0-proxy-57m | 10042 | 0.2606 | 0.2566 | 125.9900 |
| piqa | solollm-v4-2-dropout0-proxy-57m | 1000 | 0.5710 | 0.5800 | 48.1460 |
| arc_easy | solollm-v4-2-dropout0-proxy-57m | 570 | 0.3614 | 0.3070 | 17.6982 |
| arc_challenge | solollm-v4-2-dropout0-proxy-57m | 299 | 0.1639 | 0.2007 | 23.0669 |
| winogrande | solollm-v4-2-dropout0-proxy-57m | 1267 | 0.4925 | 0.4949 | 14.4167 |
