# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under the base LM. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `512`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | v2 | 10042 | 0.2682 | 0.2699 | 125.9900 |
| piqa | v2 | 1000 | 0.6070 | 0.6100 | 48.1460 |
| arc_easy | v2 | 570 | 0.4474 | 0.3789 | 17.6982 |
| arc_challenge | v2 | 299 | 0.1806 | 0.1973 | 23.0669 |
| winogrande | v2 | 1267 | 0.4949 | 0.4925 | 14.4167 |
| hellaswag | gpt2 | 10042 | 0.2855 | 0.2953 | 125.9900 |
| piqa | gpt2 | 1000 | 0.6360 | 0.6360 | 48.1460 |
| arc_easy | gpt2 | 570 | 0.4561 | 0.4035 | 17.6982 |
| arc_challenge | gpt2 | 299 | 0.1672 | 0.2207 | 23.0669 |
| winogrande | gpt2 | 1267 | 0.4854 | 0.4972 | 14.4167 |
