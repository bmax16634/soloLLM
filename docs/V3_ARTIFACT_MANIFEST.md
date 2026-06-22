# SoloLLM V3 Artifact Manifest

Date: June 19, 2026

This manifest records the canonical local artifacts after the final v3 closeout
and checkpoint cleanup. Generated datasets, model weights, logs, and eval outputs
are intentionally ignored by Git; the repository tracks the code, configs,
scripts, and result documentation needed to reproduce or audit them.

## Canonical Result

| Item | Value |
| --- | --- |
| Final best model | `v3-plus-150m-1024` |
| Params | 151,868,928 |
| Training tokens | 10,000,007,168 |
| Final held-out PPL | 24.898970785722693 |
| GPT-2 held-out PPL | 25.315036823416108 |
| Final result doc | `docs/results/v3_final_gpt2_comparison.md` |
| Final eval output | `outputs/eval_suites/v3_fresh_10b_150m_10bdata_gpt2_full_suite/` |

The smaller ablation is `v3-gpt2-scale-1024`: 123,551,232 params, 9,800,728,576
training tokens, and final held-out PPL 25.636956381474878. It is slightly
smaller than GPT-2 small and wins most external checks, but it does not beat
GPT-2 across every metric.

## Published Hugging Face Artifacts

| Artifact | URL |
| --- | --- |
| Final v3 150M base model | <https://huggingface.co/bmax16634/sologpt-v3-150m-base> |
| v3 123M smaller-model ablation | <https://huggingface.co/bmax16634/sologpt-v3-123m-base> |
| Public v3 completion demo | <https://huggingface.co/spaces/bmax16634/sologpt-v3-150m-demo> |
| Legacy v1 baseline model | <https://huggingface.co/bmax16634/sologpt-base-v1> |
| Legacy v1 baseline demo | <https://huggingface.co/spaces/bmax16634/sologpt-base-v1-demo> |

The v3 model repos include `model.safetensors`, model cards, tokenizer files,
and custom Hugging Face `AutoModelForCausalLM` remote-code wrappers. Load with
`trust_remote_code=True`.

## Dataset

| Artifact | Path |
| --- | --- |
| Data source manifest | `sologpt_v3/data_sources.yaml` |
| Builder script | `scripts/build_v3_pilot_dataset.py` |
| Notification wrapper | `scripts/build_v3_10b_dataset_notify.sh` |
| Final dataset root | `/home/bmx/_projects/soloLLM/data/v3_10b_1024` |
| Build stats | `/home/bmx/_projects/soloLLM/data/v3_10b_1024/build_stats.json` |
| Split manifest | `/home/bmx/_projects/soloLLM/data/v3_10b_1024/splits.json` |
| Combined shard directory | `/home/bmx/_projects/soloLLM/data/v3_10b_1024/chunks` |

Final split:

| Split | Shards | Tokens |
| --- | --- | ---: |
| Train | `0:1168` | 9,800,728,576 |
| Validation | `1169:1180` | 99,422,208 |
| Test | `1181:1192` | 99,848,192 |

Final source mix:

| Source | Accepted tokens | Share |
| --- | ---: | ---: |
| FineWeb-Edu `sample-10BT` | 4,000,001,532 | 40% |
| DCLM baseline | 2,500,001,319 | 25% |
| FineWeb `sample-10BT` | 1,499,997,774 | 15% |
| English Wikipedia | 999,998,937 | 10% |
| OpenWebText | 1,000,000,972 | 10% |

## Model Configs

| Config | Role |
| --- | --- |
| `sologpt_v3/config_gpt2_scale_1024.json` | 123M smaller-than-GPT-2 final ablation |
| `sologpt_v3/config_plus_150m_1024.json` | 150M final best model |
| `sologpt_v3/config_plus_180m_1024.json` | Phase gate config, not the final model |

## Final Model Artifacts

| Model | Local artifact | Notes |
| --- | --- | --- |
| v3 150M final model | `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/final_model.pt` | final inference artifact |
| v3 150M latest checkpoint | `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/checkpoints/latest.pt` | resumable checkpoint |
| v3 150M metrics | `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/metrics_summary.json` | training summary |
| v3 150M run metadata | `outputs/sologpt_v3/v3_fresh_10b_plus_150m_1024_10bdata/run_meta.json` | run config metadata |
| v3 123M final model | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/final_model.pt` | smaller-model ablation |
| v3 123M latest checkpoint | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/checkpoints/latest.pt` | resumable checkpoint |
| v3 123M metrics | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/metrics_summary.json` | training summary |
| v3 123M run metadata | `outputs/sologpt_v3/v3_fresh_10b_gpt2scale_123m_1024_10bdata/run_meta.json` | run config metadata |

The v2 diagnostic checkpoint retained for comparison is
`outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/checkpoints/latest.pt`.

## Eval Artifacts

| Eval | Output directory |
| --- | --- |
| v3 150M vs GPT-2 final suite | `outputs/eval_suites/v3_fresh_10b_150m_10bdata_gpt2_full_suite/` |
| v3 123M vs GPT-2 final suite | `outputs/eval_suites/v3_fresh_10b_123m_10bdata_gpt2_full_suite/` |
| v2 5.60B vs GPT-2 diagnostic suite | `outputs/eval_suites/v2_5p6b_gpt2_full_suite/` |

Each final eval directory contains:

- candidate held-out JSON,
- `gpt2_heldout.json`,
- `external_full_candidate_gpt2.json`,
- `multiple_choice_full_candidate_gpt2.json`,
- fixed generation JSON and generation metrics,
- Markdown reports under `reports/`,
- `suite_manifest.json`,
- `v3_eval_suite_full.log`.

## Run Scripts

| Script | Purpose |
| --- | --- |
| `scripts/build_v3_10b_dataset_notify.sh` | Build the final 10B-token dataset with notifications |
| `scripts/run_v3_full_150m_notify.sh` | Start the original 1B v3 150M run |
| `scripts/run_v3_continue_2b_150m_notify.sh` | Continue the 150M model to 2B tokens |
| `scripts/run_v3_continue_3b_150m_notify.sh` | Continue the 150M model to 3B tokens |
| `scripts/run_v3_fresh_5b_150m_10bdata_notify.sh` | Fresh 150M run on the 10B dataset to 5B tokens |
| `scripts/run_v3_continue_10b_150m_10bdata_notify.sh` | Continue the fresh 150M model to 10B tokens |
| `scripts/run_v3_fresh_10b_123m_train_eval_notify.sh` | Train and eval the final 123M model |
| `scripts/run_v3_1b_full_eval_notify.sh` | General full-suite eval wrapper for v3/v2 candidates |

## Cleanup Policy

The cleanup removed numbered recovery checkpoints matching `outputs/**/checkpoints/ckpt_*.pt`.
Those files were useful during training but are not required for the final
portfolio result after `latest.pt`, `final_model.pt`, metrics, configs, logs, and
eval outputs were retained.

Retain:

- final `latest.pt` checkpoints for v2 5.60B, v3 123M, and v3 150M,
- final `final_model.pt` files for v3 123M and v3 150M,
- `metrics_summary.json`, `config_resolved.json`, and `run_meta.json`,
- final eval suite outputs,
- the final 10B dataset until/unless it is rebuilt from the manifest.

Do not rely on old numbered checkpoint names in historical docs as current local
artifacts. They describe the training timeline; the retained checkpoint artifact
for final use is `checkpoints/latest.pt`.
