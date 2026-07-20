# External Base-LM Benchmark Results

Compact external comparison using each model's native tokenizer and a shared context window.

Raw token perplexity is tokenizer-local. Use bits-per-byte and downstream accuracies for cross-tokenizer comparisons.

- Context length: `1024`
- Device: `cuda`
- WikiText token cap: `full`
- LAMBADA example cap: `full`

| Benchmark | Model | Tokens | Bytes | BPB | PPL | Loss | Last-token acc | Last-word exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | pythia-410m | 288439 | 1287656 | 0.9791 | 20.6919 | 3.0297 | n/a | n/a |
| lambada | pythia-410m | 423382 | 1688837 | 1.2337 | 30.3015 | 3.4112 | 0.6602 | 0.5251 |
| wikitext2 | tinyllama-1.1b | 335315 | 1287656 | 0.9630 | 12.9776 | 2.5632 | n/a | n/a |
| lambada | tinyllama-1.1b | 461050 | 1688837 | 1.1828 | 20.1474 | 3.0031 | 0.8007 | 0.5880 |

LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.
