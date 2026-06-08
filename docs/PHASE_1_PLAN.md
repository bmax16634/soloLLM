# Phase 1 Plan

Phase 1 turns the Phase 0 foundation into a real architecture comparison on the RTX 3090.

Result tracking: `docs/results/phase1_50m_sanity.md`

Status: complete. `v2-modern-small` is the 300M pilot candidate.

## Goal

Prepare and sanity-test two v2 variants:

| Variant | Purpose | Target |
| --- | --- | --- |
| `v2-gpt2-parity` | Control model | 123,616,512 params, GPT-2-small scale |
| `v2-modern-small` | Main experiment | 91,654,400 params, smaller than GPT-2-small |

The question Phase 1 answered is:

> Which v2 variant is healthy enough to spend 300M tokens on?

Answer: `v2-modern-small`.

Phase 1 is not the final training run.

## Work Item 1: Add Modern Architecture Switches

Extend `sologpt_v2/model.py` without breaking the existing import:

```python
from sologpt_v2 import SoloGPT_v2
```

Required config switches:

| Key | Values | Purpose |
| --- | --- | --- |
| `norm_type` | `layernorm`, `rmsnorm` | Keep parity model GPT-like, allow modern-small to use RMSNorm. |
| `mlp_type` | `gelu`, `swiglu` | Keep parity model simple, allow modern-small to use SwiGLU. |
| `use_bias` | `true`, `false` | Let modern-small use bias-free linears. |
| `init_std` | float | GPT-style normal initialization. |
| `scale_residual_init` | bool | Scale residual projection weights by model depth. |

Keep existing behavior as the default:

- `norm_type`: `layernorm`
- `mlp_type`: `gelu`
- `use_bias`: current layer-specific defaults

Status: done.

## Work Item 2: Add Modern-Small Config

Add:

```text
sologpt_v2/config_modern_small.json
```

Target shape:

```json
{
  "variant": "modern_small",
  "vocab_size": 50257,
  "n_embd": 640,
  "n_layer": 12,
  "n_head": 10,
  "seq_length": 512,
  "max_seq_len": 512,
  "dropout": 0.1,
  "rope_base": 10000.0,
  "tie_weights": true,
  "qkv_bias": false,
  "norm_type": "rmsnorm",
  "mlp_type": "swiglu",
  "use_bias": false,
  "init_std": 0.02,
  "scale_residual_init": true
}
```

Acceptance:

- parameter count is below GPT-2-small,
- measured parameter count is 91,654,400,
- CPU tests pass.

Status: done.

## Work Item 3: Add Architecture Tests

Add tests for:

- default parity config still instantiates,
- modern-small config instantiates,
- RMSNorm path works,
- SwiGLU path works,
- tied embeddings still share storage,
- modern-small param count is below parity model,
- over-context input still raises `ValueError`,
- tiny modern-small training dry run writes artifacts.

Test command:

```bash
python -m pytest
```

Status: done for model/config coverage. GPU sanity is still pending.

## Work Item 4: Run 50M Sanity For Both Variants

Run parity control:

```bash
python -m sologpt_v2.pretrain \
  --config sologpt_v2/config.json \
  --shard-dir data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/sanity_50m_parity \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 50000000 \
  --max-eval-tokens 5000000
```

Status: done.

Run modern-small:

```bash
python -m sologpt_v2.pretrain \
  --config sologpt_v2/config_modern_small.json \
  --shard-dir data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/sanity_50m_modern_small \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 50000000 \
  --max-eval-tokens 5000000
```

Status: done.

Expected 3090 runtime:

| Variant | Expected Time |
| --- | ---: |
| `v2-gpt2-parity` | about 30-75 minutes |
| `v2-modern-small` | about 20-60 minutes |

## Work Item 5: Compare Sanity Results

Compare:

- final train loss,
- capped validation loss,
- tokens/sec average,
- GPU peak memory,
- checkpoint existence,
- resume behavior,
- loss curve shape,
- a few short generations if practical.

Status: done except generations, which are not necessary for this sanity gate.

Useful files:

```text
outputs/sologpt_v2/sanity_50m_*/metrics.jsonl
outputs/sologpt_v2/sanity_50m_*/metrics_summary.json
outputs/sologpt_v2/sanity_50m_*/checkpoints/latest.pt
```

## Phase 1 Decision

Move to the 300M pilot when:

- `python -m pytest` passes,
- both variants instantiate,
- at least one variant completes 50M tokens,
- loss is finite and trending down,
- checkpoint resume works,
- GPU peak memory is safe on the 3090.

Outcome:

- `v2-modern-small` completed 50M tokens in about 16.4 minutes.
- `v2-modern-small` reached lower train loss and lower capped validation perplexity than `v2-gpt2-parity`.
- `v2-modern-small` used fewer parameters and produced smaller artifacts.
- `v2-modern-small` should be the fresh 300M pilot run.

Original decision rule:

- If `v2-modern-small` is close to parity loss and meaningfully faster, make it the main 300M pilot.
- If `v2-modern-small` is clearly worse or unstable, use `v2-gpt2-parity` for the 300M pilot and keep modern-small as an experiment.
- If both are unstable, fix architecture/training before spending more GPU time.
