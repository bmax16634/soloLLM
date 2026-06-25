# External Base-LM Benchmark Results

Compact external comparison using each model's native tokenizer and a shared context window.

Raw token perplexity is tokenizer-local. Use bits-per-byte and downstream accuracies for cross-tokenizer comparisons.

- Context length: `1024`
- Device: `cuda`
- WikiText token cap: `full`
- LAMBADA example cap: `full`

| Benchmark | Model | Tokens | Bytes | BPB | PPL | Loss | Last-token acc | Last-word exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | solollm-v3-150m | 285897 | 1287656 | 1.1909 | 41.1814 | 3.7180 | n/a | n/a |
| lambada | solollm-v3-150m | 434956 | 1688837 | 1.3247 | 35.3472 | 3.5652 | 0.4743 | 0.3307 |
| wikitext2 | gpt2 | 285897 | 1287656 | 1.2216 | 45.3195 | 3.8137 | n/a | n/a |
| lambada | gpt2 | 434956 | 1688837 | 1.3764 | 40.6243 | 3.7044 | 0.4667 | 0.3260 |
| wikitext2 | distilgpt2 | 285897 | 1287656 | 1.4104 | 81.7089 | 4.4032 | n/a | n/a |
| lambada | distilgpt2 | 434956 | 1688837 | 1.5041 | 57.2903 | 4.0481 | 0.4087 | 0.2498 |
| wikitext2 | pythia-70m | 288439 | 1287656 | inf | inf | inf | n/a | n/a |
| lambada | pythia-70m | 423382 | 1688837 | 6.7967 | 145015763.0386 | 18.7924 | 0.4188 | 0.1908 |
| wikitext2 | pythia-160m | 288439 | 1287656 | 2.0182 | 515.4586 | 6.2451 | n/a | n/a |
| lambada | pythia-160m | 423382 | 1688837 | 3.0843 | 5053.1609 | 8.5278 | 0.5236 | 0.3433 |
| wikitext2 | smollm-135m | 304540 | 1287656 | 1.1849 | 32.2262 | 3.4728 | n/a | n/a |
| lambada | smollm-135m | 425095 | 1688837 | 1.3489 | 41.0451 | 3.7147 | 0.5447 | 0.3771 |
| wikitext2 | smollm2-135m | 304540 | 1287656 | 1.1071 | 25.6502 | 3.2446 | n/a | n/a |
| lambada | smollm2-135m | 425095 | 1688837 | 1.2616 | 32.2695 | 3.4741 | 0.5833 | 0.4267 |
| wikitext2 | smollm2-360m | 304540 | 1287656 | 1.0251 | 20.1752 | 3.0045 | n/a | n/a |
| lambada | smollm2-360m | 425095 | 1688837 | 1.2004 | 27.2678 | 3.3057 | 0.6649 | 0.5325 |

LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.
