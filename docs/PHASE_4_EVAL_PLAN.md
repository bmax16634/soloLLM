# Phase 4 Final Evaluation Plan

Phase 4 is the final comparison package for SoloLLM v2. It should answer more than "what is the perplexity?"

## Goal

Compare `v2-modern-small` against `sologpt_v1` and GPT-2-small in a way that is fair, reproducible, and useful for a portfolio reviewer.

The headline question:

> Can a smaller modern model trained from scratch on a single RTX 3090 approach or beat GPT-2-small on fair held-out evaluation?

## Models To Compare

| Model | Purpose |
| --- | --- |
| `sologpt_v1` | Published baseline / rough prototype |
| `v2-gpt2-parity` | GPT-2-small-scale architecture control, if using its available checkpoint |
| `v2-modern-small` 300M | Pilot checkpoint for scaling trend |
| `v2-modern-small` final | Main result |
| GPT-2-small | External baseline |

## Quantitative Evaluations

Primary:

- Held-out test perplexity on shards `58:60`.
- Use the same tokenized data, same next-token loss computation, and same batching path for v1, v2, and GPT-2.
- Report total loss, perplexity, token count, parameter count, train tokens, and training hardware.

Robustness checks:

- Per-shard perplexity table for each test shard.
- Bootstrap or chunk-level confidence intervals over held-out chunks.
- Short-context vs full-512-token-window perplexity, if easy to add.
- Compare validation trend from training with final test perplexity to check for overfitting.

Efficiency metrics:

- Params.
- Training tokens.
- Wall-clock training time.
- Peak VRAM.
- Tokens/sec.
- Checkpoint size.
- Hardware: single RTX 3090.

## Qualitative Evaluations

Use a fixed prompt suite, deterministic seeds, and the same sampling settings across models.

Prompt categories:

- Factual-style completions.
- Story continuation.
- Code-ish / structured text.
- Reasoning-looking prompts.
- Repetition/stability stress prompts.
- Domain prompts related to the training distribution.

For each model:

- Save raw completions.
- Include 3-5 representative successes.
- Include 3-5 representative failures.
- Do not cherry-pick only good samples.

## Optional External Benchmarks

If time allows, add a small base-LM benchmark set through `lm-eval-harness` or a lightweight local equivalent:

- LAMBADA or WikiText-style perplexity.
- HellaSwag.
- PIQA.
- ARC-Easy.

These are optional because the project model is a small base LM trained from scratch, not an instruction model. The must-have comparison is still the fair held-out perplexity plus fixed generation suite.

## Result Table

Status: complete for the official 3B-token `v2-modern-small` checkpoint and the durable 5.60B stretch checkpoint. The final README includes this compact full held-out table:

| Model | Params | Train tokens | Test PPL | Peak VRAM | Hardware | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| v1 | 203.75M | 9.03B declared target, actual unverified | 30.48 | n/a | RTX 3090-era project baseline | rough baseline, published HF checkpoint |
| v2-modern-small final | 91.65M | 3.00B confirmed | 26.26 | 10.48GB | single RTX 3090 | headline 3B-token result |
| v2-modern-small stretch | 91.65M | 5.60B checkpoint metadata | 25.56 | 10.48GB-class run | single RTX 3090 | best official v2 checkpoint |
| GPT-2-small | 124.44M | external | 25.32 | n/a | external | reference baseline |

The durable 5.60B stretch checkpoint also received the full held-out eval:

| Model | Params | Checkpoint / train tokens | Eval tokens | Test loss | Test PPL | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| v2-modern-small stretch | 91.65M | `latest.pt`, metadata `5,600,206,848` tokens | 331,353,862 | 3.2408 | 25.56 | Durable 5.60B checkpoint |
| GPT-2-small | 124.44M | external | 331,353,862 | 3.2314 | 25.32 | reference baseline |

The stretch result is now an official full held-out comparison. It improves on the 3B checkpoint by about 2.7% lower perplexity and remains about 0.95% higher perplexity than GPT-2 small.

The optional robust comparison layer is also complete for the 5.60B checkpoint:

| Check | v2-modern-small 5.60B | GPT-2 small | Notes |
| --- | ---: | ---: | --- |
| Fixed prompts corpus distinct-2 | 0.7512 | 0.7836 | GPT-2 slightly higher diversity |
| Fixed prompts repeated bigram fraction | 0.2015 | 0.1715 | GPT-2 slightly less repetitive |
| WikiText-2 PPL | 83.24 | 49.34 | 249,511 scored tokens, shared 512 context |
| LAMBADA PPL | 62.46 | 42.37 | 1,000-example cap, shared 512 context |
| LAMBADA last-token accuracy | 44.3% | 46.6% | lightweight approximation |
| LAMBADA last-word greedy exact | 28.8% | 31.7% | lightweight approximation |

Conclusion: v2 is close to GPT-2 on the project held-out split, but GPT-2 remains more robust on external corpora and generation stability checks.

The v3 plan in `docs/V3_PLAN.md` treats this suite as the frozen comparability baseline and expands it for any public claim that v3 beats GPT-2 across the board.

## Decision Logic

If v2 beats v1 but not GPT-2:

- The project still succeeds as a strong v1-to-v2 engineering improvement.
- Decide whether to continue with more tokens, improve data, or increase model size based on train/validation/test behavior.

If v2 approaches GPT-2:

- Emphasize smaller-parameter, consumer-hardware training.
- Use the full-evaluated 5.60B stretch checkpoint as the strongest v2 perplexity result, while noting it still does not beat GPT-2 and is weaker on external corpora.

If v2 beats GPT-2 on the fixed test:

- Treat it as the headline result, but still report limitations and sample failures.

## Artifacts To Save

- `docs/results/final_eval.md`
- JSON outputs from `eval/eval.py`
- Prompt suite file
- Raw generations
- Generation metrics JSON/Markdown
- External benchmark JSON/Markdown
- Plots for loss/validation trend
- README result table
