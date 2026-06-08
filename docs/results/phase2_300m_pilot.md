# Phase 2 300M Pilot Results

## v2-modern-small

Status: complete

Final run directory:

```text
outputs/sologpt_v2/pilot_300m_modern_small_resume100m
```

Partial/interrupted run directory:

```text
outputs/sologpt_v2/pilot_300m_modern_small
```

## Run History

The first fresh 300M pilot reached `183.5M` tokens, but the interactive PTY process was killed before the `200M` checkpoint. The last durable checkpoint was `ckpt_100M.pt`, so the final run resumed cleanly from that checkpoint into a new output directory.

Resume command:

```bash
setsid scripts/run_v2_modern_small_300m_resume.sh >/dev/null 2>&1 < /dev/null &
```

Wrapper script:

```text
scripts/run_v2_modern_small_300m_resume.sh
```

Final resume window:

- Started: June 6, 2026, 1:58 PM AZ
- Finished: June 6, 2026, 3:03 PM AZ
- Completion notification: Ariya reminder id `10eac81a-121b-4fe8-9d65-03dc66eb2944`; user confirmed receipt.

## Summary

| Metric | Value |
| --- | ---: |
| Variant | `v2-modern-small` |
| Parameters | 91,654,400 |
| Final tokens | 300,007,424 |
| Optimizer steps | 9,156 |
| Final train loss | 3.54837965965271 |
| 300M capped validation loss | 3.76004081029518 |
| 300M capped validation PPL | 42.950178752470094 |
| Validation tokens | 10,007,424 |
| Resume wall time | 3,881.53 sec / 64.7 min |
| Effective resume throughput | about 51,500 tok/s |
| Peak GPU memory | 10.48GB |

Notes:

- The generated `metrics_summary.json` reports `tokens_per_sec_avg=77,291`, but that is inflated for resumed runs because it divides total tokens including the `100M` checkpoint baseline by only the resume wall time.
- The effective resume throughput is computed as `(300,007,424 - 100,007,936) / 3,881.53`, which is about `51.5k tok/s`.
- GPU utilization stayed near 100% during training, and GPU memory stayed far below the RTX 3090's 24GB limit.
- GPU was idle after completion.

## Validation Trend

| Token point | Source | Validation loss | Validation PPL | Validation tokens |
| ---: | --- | ---: | ---: | ---: |
| 100M | original pilot auto eval | 4.222912514521406 | 68.2319216359067 | 10,007,424 |
| 200M | resumed pilot auto eval | 3.888195095693364 | 48.82268667409195 | 10,007,424 |
| 300M | manual final eval | 3.76004081029518 | 42.950178752470094 | 10,007,424 |

The final manual eval was added because the resumed training loop stopped just under the next `eval_every_tokens=100M` cadence after the `200M` eval. It is saved at:

```text
outputs/sologpt_v2/pilot_300m_modern_small_resume100m/final_eval_300m.json
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Final checkpoint | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/checkpoints/latest.pt` |
| 200M checkpoint | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/checkpoints/ckpt_200M.pt` |
| Final model weights | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/final_model.pt` |
| Metrics JSONL | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/metrics.jsonl` |
| Training summary | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/metrics_summary.json` |
| Final eval | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/final_eval_300m.json` |
| Resume log | `outputs/sologpt_v2/pilot_300m_modern_small_resume100m/resume_300m.log` |

## Decision

Phase 2 passes.

`v2-modern-small` is healthy enough to promote to the final training phase. The validation trend improved from `68.23` PPL at `100M` to `42.95` PPL at `300M`, training loss continued downward, memory stayed around `10.5GB`, and checkpoint/resume worked.

Before the final 3B-4B token run, fix two logging issues:

- resumed-run throughput should report tokens processed during the current run, not total resumed tokens divided by resume wall time;
- final validation should run when training stops near an eval boundary, even if the final token count is a few thousand tokens short of `eval_every_tokens`.

Recommended next phase: run a longer `v2-modern-small` training job, then compare held-out perplexity against GPT-2-small using the shared eval path.
