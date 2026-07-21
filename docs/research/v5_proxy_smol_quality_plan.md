# V5 Smol-Quality Proxy Plan

Date: July 21, 2026

## Objective

Run a cheap v5 proxy that tests whether SoloLLM can move toward modern
SmolLM2-style small-model quality before launching another multi-day 123M or
150M training run.

The goal is not to beat GPT-2 again. The gate is whether a 60M proxy improves
the broad quality profile that matters for the v5 baseline board:

- WikiText BPB,
- LAMBADA BPB,
- LAMBADA token and word continuation accuracy,
- normalized multiple-choice average,
- no obvious regression in local usability.

## Experiment

| Item | Value |
| --- | --- |
| Config | `sologpt_v5/config_proxy_smol_quality_60m_1024.json` |
| Data manifest | `sologpt_v5/data_mix_smol_quality_1b.yaml` |
| Target tokens | `1B` |
| Context | `1024` |
| Model shape | v4 proxy shape, about `57M` params |
| Dropout | `0.0` |
| Checkpoint cadence | `250M` tokens |
| Eval cadence | `100M` tokens |
| Launcher | `scripts/run_v5_proxy_smol_quality_1b_train_eval_notify.sh` |

## Data Hypothesis

V4 established the broad-LM versus reasoning/MC tradeoff. V5 should test a
more SmolLM2-style mix without going fully reasoning-heavy:

- keep FineWeb-Edu as the anchor,
- add more SmolLM educational-web and synthetic educational prose,
- keep DCLM/Wikipedia for broad generalization,
- keep FineMath below the v4.1b reasoning-heavy level,
- shrink generic FineWeb/OpenWebText to small distribution anchors.

## Gate

Scale only if this proxy improves broad BPB and LAMBADA continuation while
holding or improving MC average versus the v4.2 proxy. A mixed result should
lead to another cheap proxy, not a full 123M/150M run.

## Outputs

- Train run: `outputs/sologpt_v5/proxy_smol_quality_60m_1b/`
- Eval suite: `outputs/eval_suites/v5_proxy_smol_quality_60m_1b/`
- Summary: `docs/results/v5_proxy_smol_quality_60m_1b_summary.md`
