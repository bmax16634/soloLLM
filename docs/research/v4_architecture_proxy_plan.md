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
