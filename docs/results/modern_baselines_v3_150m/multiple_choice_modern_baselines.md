# Multiple-Choice Base-LM Benchmark Results

Choices are scored by conditional log-likelihood under each base LM's native tokenizer. `accuracy_norm` uses average log-probability per continuation token to reduce length bias.

- Context length: `1024`
- Max examples: `full validation split`

| Benchmark | Model | Examples | Accuracy | Accuracy norm | Avg choice tokens |
|---|---|---:|---:|---:|---:|
| hellaswag | solollm-v3-150m | 10042 | 0.2899 | 0.3008 | 125.9900 |
| piqa | solollm-v3-150m | 1000 | 0.6430 | 0.6270 | 48.1460 |
| arc_easy | solollm-v3-150m | 570 | 0.5246 | 0.4456 | 17.6982 |
| arc_challenge | solollm-v3-150m | 299 | 0.2174 | 0.2508 | 23.0669 |
| winogrande | solollm-v3-150m | 1267 | 0.5130 | 0.5114 | 14.4167 |
| hellaswag | gpt2 | 10042 | 0.2855 | 0.2953 | 125.9900 |
| piqa | gpt2 | 1000 | 0.6360 | 0.6360 | 48.1460 |
| arc_easy | gpt2 | 570 | 0.4561 | 0.4035 | 17.6982 |
| arc_challenge | gpt2 | 299 | 0.1672 | 0.2207 | 23.0669 |
| winogrande | gpt2 | 1267 | 0.4854 | 0.4972 | 14.4167 |
| hellaswag | distilgpt2 | 10042 | 0.2670 | 0.2633 | 125.9900 |
| piqa | distilgpt2 | 1000 | 0.6050 | 0.6020 | 48.1460 |
| arc_easy | distilgpt2 | 570 | 0.4368 | 0.3579 | 17.6982 |
| arc_challenge | distilgpt2 | 299 | 0.1906 | 0.2040 | 23.0669 |
| winogrande | distilgpt2 | 1267 | 0.4822 | 0.4949 | 14.4167 |
| hellaswag | pythia-70m | 10042 | 0.2636 | 0.2639 | 127.1380 |
| piqa | pythia-70m | 1000 | 0.5980 | 0.5960 | 47.9040 |
| arc_easy | pythia-70m | 570 | 0.3825 | 0.3439 | 17.6088 |
| arc_challenge | pythia-70m | 299 | 0.1973 | 0.2241 | 23.0435 |
| winogrande | pythia-70m | 1267 | 0.4917 | 0.4957 | 14.8611 |
| hellaswag | pythia-160m | 10042 | 0.2834 | 0.2972 | 127.1380 |
| piqa | pythia-160m | 1000 | 0.6280 | 0.6200 | 47.9040 |
| arc_easy | pythia-160m | 570 | 0.4439 | 0.3825 | 17.6088 |
| arc_challenge | pythia-160m | 299 | 0.1973 | 0.2241 | 23.0435 |
| winogrande | pythia-160m | 1267 | 0.4972 | 0.4980 | 14.8611 |
| hellaswag | smollm-135m | 10042 | 0.3364 | 0.4032 | 126.1840 |
| piqa | smollm-135m | 1000 | 0.6630 | 0.6560 | 48.0210 |
| arc_easy | smollm-135m | 570 | 0.6123 | 0.5491 | 17.0807 |
| arc_challenge | smollm-135m | 299 | 0.2776 | 0.3043 | 22.6421 |
| winogrande | smollm-135m | 1267 | 0.5138 | 0.5107 | 14.8950 |
| hellaswag | smollm2-135m | 10042 | 0.3420 | 0.4109 | 126.1840 |
| piqa | smollm2-135m | 1000 | 0.6820 | 0.6710 | 48.0210 |
| arc_easy | smollm2-135m | 570 | 0.6211 | 0.5667 | 17.0807 |
| arc_challenge | smollm2-135m | 299 | 0.2642 | 0.3244 | 22.6421 |
| winogrande | smollm2-135m | 1267 | 0.5059 | 0.5162 | 14.8950 |
| hellaswag | smollm2-360m | 10042 | 0.4106 | 0.5342 | 126.1840 |
| piqa | smollm2-360m | 1000 | 0.7100 | 0.7060 | 48.0210 |
| arc_easy | smollm2-360m | 570 | 0.6912 | 0.6842 | 17.0807 |
| arc_challenge | smollm2-360m | 299 | 0.3244 | 0.3813 | 22.6421 |
| winogrande | smollm2-360m | 1267 | 0.5146 | 0.5328 | 14.8950 |
