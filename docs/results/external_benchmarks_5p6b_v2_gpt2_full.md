# External Base-LM Benchmark Results

Compact external comparison using the GPT-2 tokenizer and a shared context window.

- Context length: `512`
- Device: `cuda`
- WikiText token cap: `full`
- LAMBADA example cap: `full`

| Benchmark | Model | Tokens | PPL | Loss | Last-token acc | Last-word exact |
|---|---|---:|---:|---:|---:|---:|
| wikitext2 | v2 | 285618 | 84.8120 | 4.4404 | n/a | n/a |
| lambada | v2 | 434531 | 63.0317 | 4.1436 | 0.4430 | 0.2909 |
| wikitext2 | gpt2 | 285618 | 49.8590 | 3.9092 | n/a | n/a |
| lambada | gpt2 | 434531 | 42.2566 | 3.7438 | 0.4667 | 0.3260 |

LAMBADA last-token and last-word scores are lightweight approximations for this project report. They are useful for relative comparison, but they are not a replacement for a full lm-eval-harness run.
