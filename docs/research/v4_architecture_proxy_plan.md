# V4 Architecture And Training-Recipe Proxy Plan

The data-mix proxy loop exposed a real broad-LM versus MC/reasoning tradeoff,
but no blended data-only proxy earned a full 150M/10B scale-up. The next
controlled variable is the architecture/training recipe.

## Current Architecture

The current SoloLLM v3/v4 model is already a modern GPT-style decoder:

- RoPE positional encoding
- RMSNorm
- SwiGLU MLP
- tied input/output embeddings
- bias-free linear projections
- causal scaled-dot-product attention
- GPT-2 tokenizer
- 1024-token context

Full 150M shape:

- `n_layer=16`
- `n_embd=768`
- `n_head=12`
- `dropout=0.1`
- no grouped-query attention in the full 150M config

57M proxy shape:

- `n_layer=10`
- `n_embd=512`
- `n_head=8`
- `dropout=0.1`

## Completed Architecture Signal

The deeper/narrower SmolLM2-style 57M proxy did not justify replacing the
wider v3-style proxy shape by itself.

## V4.2 Proxy

First training-recipe test:

- Data: `v4_1b_blend_300m_1024`
- Model: same 57M v3-style proxy
- Change: `dropout=0.1` to `dropout=0.0`
- Config: `sologpt_v4/config_proxy_v4_2_dropout0_57m_1024.json`
- Launcher: `scripts/run_v4_2_dropout0_proxy_eval_notify.sh`

Decision rule:

- Scale only if dropout `0.0` improves proxy validation and does not worsen the
  modern broad-LM/MC tradeoff.
- If dropout `0.0` overfits or weakens external metrics, try `dropout=0.05`
  before changing the transformer block shape again.

## V4.2 Result

| Proxy | Val PPL | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|---:|
| v4.1b dropout0.1 | 54.46 | 149.37 | 68.55 | 0.2381 | 0.3613 |
| v4.2 dropout0.0 | 51.28 | 149.84 | 65.44 | 0.2366 | 0.3678 |

Read: dropout `0.0` improves validation, LAMBADA PPL, and MC average, while
leaving WikiText roughly flat/slightly worse. This is a stronger signal than
the previous data-only interpolation.

Next proxy: run the same dropout `0.0` recipe on the broader v4.1 blend data.
If it keeps v4.1's better WikiText while improving MC, that becomes the first
credible full-scale candidate.

## V4.2 On V4.1 Blend Follow-Up

This follow-up tests whether the dropout-0 gain transfers to the broader data
mix that had the best WikiText/LAMBADA proxy behavior.

- Data: `v4_1_blend_300m_1024`
- Model: same 57M v3-style proxy
- Change versus v4.1: `dropout=0.1` to `dropout=0.0`
- Config: `sologpt_v4/config_proxy_v4_2_dropout0_v4_1_blend_57m_1024.json`
- Launcher: `scripts/run_v4_2_dropout0_v4_1_blend_proxy_eval_notify.sh`

Decision rule: scale only if it keeps v4.1's broad external advantage while
materially improving validation and/or MC.

## V4.2 On V4.1 Blend Result

| Proxy | Val PPL | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|---:|
| v4.1 blend dropout0.1 | 58.45 | 144.80 | 66.93 | 0.2408 | 0.3601 |
| v4.1 blend dropout0.0 | 54.98 | 141.55 | 64.54 | 0.2420 | 0.3705 |

Decision: scale this recipe. Dropout `0.0` on the broader v4.1 blend data is
the first proxy to improve both broad external metrics and MC. The next step is
to build the same mix at 10B tokens and train a fresh 150M model with dropout
`0.0`.

## Lockbox Verification

Before building the full 10B-token dataset, run a separate sanity check that was
not used for the proxy search decision:

- project held-out shards `58:60`,
- fixed sample cap of `512` batches per shard,
- fixed prompt generations,
- generation repetition/diversity metrics.

Launcher: `scripts/run_v4_2_dropout0_v4_1_blend_lockbox_notify.sh`

Result:

| Model | Held-out PPL | Repeated bigrams | Bad loops |
|---|---:|---:|---:|
| v4.2 dropout0 v4.1 blend proxy 57M | 67.57 | 0.2567 | 3 |
| GPT-2 small | 25.56 | 0.1715 | 2 |

Read: the proxy is not itself competitive with GPT-2. The useful signal is that
it is the best scale candidate from the controlled proxy search, not that the
57M proxy is a final-quality model.

## Full-Scale V4.2 Dataset

The full run used a fresh 10B-token dataset with the same v4.1 blend that won
the dropout proxy test.

- Manifest: `sologpt_v4/data_mix_v4_2_dropout0_v4_1_blend_10b.yaml`
- Build launcher: `scripts/build_v4_2_dropout0_v4_1_blend_10b_dataset_notify.sh`
- Output dataset: `data/v4_2_dropout0_v4_1_blend_10b_1024`

This dataset was for a fresh 150M dropout-0 run. Do not continue from the 57M
proxy checkpoint and do not train the 150M model by looping the 300M proxy
dataset.

Split metadata:

| Split | Shards | Tokens |
|---|---:|---:|
| Train | 1169 | 9,800,571,904 |
| Validation | 12 | 99,416,064 |
| Test | 12 | 100,012,032 |

## Full-Scale V4.2 Result

The 150M run completed after recovering from a power outage at the 8B checkpoint.
The resume path continued on unseen training shards and finished at
`9,800,572,928` total tokens.

| Item | Value |
|---|---:|
| Parameters | 151,868,928 |
| Dropout | 0.0 |
| Final train loss | 3.0599 |
| Best internal validation PPL | 19.11 |
| Full held-out PPL | 24.3035 |
| Full held-out loss | 3.1906 |

External and multiple-choice comparison:

| Benchmark | GPT-2 small | v3 150M | v4 reasoning 150M | v4.2 blend 150M |
|---|---:|---:|---:|---:|
| Project held-out PPL | 25.32 | 24.90 | 27.38 | 24.30 |
| WikiText-2 PPL | 45.32 | 41.18 | 43.59 | 41.75 |
| LAMBADA PPL | 40.62 | 35.35 | 38.46 | 34.06 |
| LAMBADA last-token acc | 46.67% | 47.40% | 47.09% | 47.27% |
| LAMBADA last-word exact | 32.60% | 33.07% | 32.91% | 33.05% |
| MC avg acc norm | 41.05% | 42.71% | 43.52% | 43.61% |

Multiple-choice normalized detail:

| Benchmark | GPT-2 small | v4.2 blend 150M |
|---|---:|---:|
| HellaSwag | 29.53% | 32.99% |
| PIQA | 63.60% | 65.50% |
| ARC-Easy | 40.35% | 45.09% |
| ARC-Challenge | 22.07% | 23.08% |
| WinoGrande | 49.72% | 51.38% |

Decision: full-scale v4.2 passed the scale test. It is now the best
quantitative SoloLLM checkpoint and the first v4 run to beat GPT-2 small across
the fixed quantitative suite. The caveat is that it remains a larger model than
GPT-2 small, so the smaller-than-GPT-2 hypothesis is still open.

Full report: `docs/results/v4_2_dropout0_v4_1_blend_150m_10b.md`.
