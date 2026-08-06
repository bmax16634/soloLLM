# SoloLLM Data

SoloLLM training data under `data/` is generated tokenized shard output, not source code. It is intentionally ignored by git because it is large binary data that can be rebuilt from tracked recipes.

## Current local generated data

At cleanup time, local `data/` contained these generated shard sets:

| Local path | Approx size | Meaning | Keep policy |
|---|---:|---|---|
| `data/solollm-1.0-broad-blend-10b-1024/` | 21.8 GB | Completed 10B-token broad-blend dataset used by current SoloLLM 1.0/303M workflow naming. | Rebuildable; no git/LFS. |
| `data/solollm-1.0-broad-blend-10b-1024.partial-20260801T0608Z/` | 20.3 GB | Interrupted/partial 10B-token broad-blend dataset build. | Disposable after confirming no active run depends on it. |
| `data/v5_high_quality_web_1b_1024/` | 8.0 GB | Generated 1B-token high-quality-web proxy dataset. | Rebuildable; no git/LFS. |
| `data/v5_math_edu_1b_1024/` | 8.0 GB | Generated 1B-token math/education proxy dataset. | Rebuildable; no git/LFS. |

The files are mostly `train/shard_*.pt`, `val/shard_*.pt`, split metadata, and build stats.

## Published model data recipe

The published Hugging Face models are:

- `bmax16634/solollm-1.0-123m-base`
- `bmax16634/solollm-1.0-152m-base`

Both published models were trained on the SoloLLM 1.0 broad-blend 10B-token recipe.

Tracked recipe files:

- `data_mixes/solollm-1.0-broad-blend-10b.yaml`
- `scripts/build_v4_2_dropout0_v4_1_blend_10b_dataset_notify.sh`
- `scripts/build_v3_pilot_dataset.py`
- `configs/solollm-1.0-123m-base.json`
- `configs/solollm-1.0-152m-base.json`
- `scripts/run_v4_2_dropout0_v4_1_blend_10b_123m_train_eval_notify.sh`
- `scripts/run_v4_2_dropout0_v4_1_blend_10b_150m_train_eval_notify.sh`

Main source datasets in the 10B-token recipe:

- `HuggingFaceFW/fineweb-edu`, config `sample-10BT`
- `mlfoundations/dclm-baseline-1.0-parquet`
- `HuggingFaceFW/fineweb`, config `sample-10BT`
- `Skylion007/openwebtext`, config `plain_text`
- `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2`
- `HuggingFaceTB/finemath`, config `finemath-4plus`
- `HuggingFaceTB/smollm-corpus`, config `fineweb-edu-dedup`
- `wikimedia/wikipedia`, config `20231101.en`

Recipe settings:

- tokenizer: GPT-2
- sequence length: 1024
- target tokens: 10,000,000,000
- shard tokens: 8,388,608
- split: 98% train, 1% val, 1% test
- seed: 2141

## Rebuild commands

Rebuild the dataset path expected by the published 123M/152M training configs:

```bash
cd /home/bmx/_projects/soloLLM
bash scripts/build_v4_2_dropout0_v4_1_blend_10b_dataset_notify.sh
```

That script writes by default to:

```text
data/v4_2_dropout0_v4_1_blend_10b_1024/
```

The 123M/152M configs currently reference:

```text
data/v4_2_dropout0_v4_1_blend_10b_1024/chunks
```

The newer 303M workflow uses the same tracked manifest but writes to the newer name:

```text
data/solollm-1.0-broad-blend-10b-1024/
```

via:

```bash
cd /home/bmx/_projects/soloLLM
bash scripts/run_solollm_1_0_303m_base_10b_train_eval_notify.sh
```

## Naming note

There are two names for the same broad-blend 10B dataset family:

| Older training-config name | Newer SoloLLM 1.0 name |
|---|---|
| `v4_2_dropout0_v4_1_blend_10b_1024` | `solollm-1.0-broad-blend-10b-1024` |

Before deleting or rebuilding data, use the script/config defaults intentionally. If a run expects the old name, either rebuild to the old path or set `DATA_DIR`/`SHARD_DIR` explicitly.

## Storage policy

- Do not track `data/` in git.
- Do not use Git LFS for generated shard sets by default.
- Prefer rebuild-from-recipe for generated tokenized datasets.
- Use fileserver artifacts only when an exact generated dataset snapshot is needed and cannot be cheaply regenerated.
- The interrupted `.partial-*` dataset folders are disposable unless an active run is explicitly using them.

