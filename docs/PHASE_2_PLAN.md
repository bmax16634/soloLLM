# Phase 2 Plan

Phase 2 is the 300M-token pilot for `v2-modern-small`.

Status: complete. Results are recorded in `docs/results/phase2_300m_pilot.md`.

## Goal

Decide whether `v2-modern-small` is healthy enough for the final 3B-4B token training run.

Phase 2 should be a fresh run, not a resume from the 50M sanity checkpoint. The extra cost is small, and the run artifact is cleaner for reporting.

Actual run note: the pilot did start fresh, but the first process was interrupted after `183.5M` tokens before the `200M` checkpoint. The final reported artifact resumed from the fresh pilot's `100M` checkpoint into a clean result directory. See `docs/results/phase2_300m_pilot.md`.

## Candidate

| Field | Value |
| --- | --- |
| Variant | `v2-modern-small` |
| Params | 91,654,400 |
| Architecture | RoPE, RMSNorm, SwiGLU, tied embeddings |
| Hardware | Single RTX 3090 |
| Token target | 300,000,000 |
| Validation cap | 10,000,000 tokens per eval |
| Checkpoint cadence | 100,000,000 tokens |
| Eval cadence | 100,000,000 tokens |

## Command

```bash
nice -n 10 ionice -c2 -n7 python -m sologpt_v2.pretrain \
  --config sologpt_v2/config_modern_small.json \
  --shard-dir /home/bmx/_projects/soloLLM/data/tokenized_chunks \
  --output-dir outputs/sologpt_v2/pilot_300m_modern_small \
  --train-shards 0:54 \
  --val-shards 55:57 \
  --max-tokens 300000000 \
  --max-eval-tokens 10000000 \
  --eval-every-tokens 100000000 \
  --tokens-per-checkpoint 100000000 \
  --no-progress
```

## Expected Runtime

Based on the 50M sanity run:

- Training throughput: about 50,924 tok/s.
- Training-only time: about 98 minutes.
- With validation and checkpoint writes: about 2-3 hours.

Expected peak VRAM:

- about 10.5GB unless batch size changes.

Expected disk output:

- latest checkpoint,
- 100M/200M/300M checkpoints,
- final model,
- metrics files,
- roughly 4GB-5GB total if all periodic checkpoints are kept.

## What To Inspect

Primary metrics:

- final train loss,
- validation loss at 100M, 200M, and 300M,
- validation perplexity trend,
- tokens/sec,
- GPU peak memory,
- checkpoint health.

Secondary checks:

- no NaNs,
- no exploding gradient norm,
- LR schedule transitions cleanly after 50M warmup,
- Ariya/DLPC remains usable enough during the run.

## Decision Criteria

Move to the final 3B-4B run if:

- training loss continues downward,
- validation loss improves or remains healthy,
- throughput remains near the 50M result,
- GPU memory stays comfortably below 24GB,
- checkpoints and resume metadata are valid.

Do not start the final run if:

- validation loss degrades sharply,
- gradient norms become unstable,
- throughput collapses,
- GPU memory unexpectedly climbs,
- output artifacts are incomplete.

## Result Location

The pilot result is recorded in:

```text
docs/results/phase2_300m_pilot.md
```
