# External Base-LM Benchmark Results

Compact external comparison using the GPT-2 tokenizer and a shared context window.

- Context length: `512`
- Device: `cuda`
- WikiText token cap: `250000`
- LAMBADA example cap: `1000`

| Benchmark | Model | Tokens | PPL | Loss | Last-token acc | Last-word exact |
|---|---|---:|---:|---:|---:|---:|
| wikitext2 | v2 | 249511 | 83.2448 | 4.4218 | n/a | n/a |
| lambada | v2 | 84474 | 62.4619 | 4.1346 | 0.4430 | 0.2880 |
| wikitext2 | gpt2 | 249511 | 49.3430 | 3.8988 | n/a | n/a |
| lambada | gpt2 | 84474 | 42.3745 | 3.7465 | 0.4660 | 0.3170 |

LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.
