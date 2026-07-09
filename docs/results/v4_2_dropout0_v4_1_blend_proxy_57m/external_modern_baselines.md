# External Base-LM Benchmark Results

Compact external comparison using each model's native tokenizer and a shared context window.

Raw token perplexity is tokenizer-local. Use bits-per-byte and downstream accuracies for cross-tokenizer comparisons.

- Context length: `1024`
- Device: `cuda`
- WikiText token cap: `full`
- LAMBADA example cap: `full`

| Benchmark | Model | Tokens | Bytes | BPB | PPL | Loss | Last-token acc | Last-word exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | solollm-v4-2-dropout0-v4-1-blend-proxy-57m | 285897 | 1287656 | 1.5864 | 141.5518 | 4.9527 | n/a | n/a |
| lambada | solollm-v4-2-dropout0-v4-1-blend-proxy-57m | 434956 | 1688837 | 1.5484 | 64.5379 | 4.1673 | 0.2420 | 0.0604 |

LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.
