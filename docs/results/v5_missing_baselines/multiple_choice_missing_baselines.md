# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under each base LM's native tokenizer. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `1024`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | pythia-410m | 10042 | 0.3353 | 0.3936 | 127.1380 |
| piqa | pythia-410m | 1000 | 0.6760 | 0.6840 | 47.9040 |
| arc_easy | pythia-410m | 570 | 0.5228 | 0.4421 | 17.6088 |
| arc_challenge | pythia-410m | 299 | 0.2274 | 0.2308 | 23.0435 |
| winogrande | pythia-410m | 1267 | 0.4957 | 0.5083 | 14.8611 |
| hellaswag | tinyllama-1.1b | 10042 | 0.4330 | 0.5683 | 140.8787 |
| piqa | tinyllama-1.1b | 1000 | 0.7420 | 0.7260 | 55.9800 |
| arc_easy | tinyllama-1.1b | 570 | 0.5754 | 0.4684 | 21.5947 |
| arc_challenge | tinyllama-1.1b | 299 | 0.2910 | 0.2943 | 27.2676 |
| winogrande | tinyllama-1.1b | 1267 | 0.5328 | 0.5233 | 15.7885 |
