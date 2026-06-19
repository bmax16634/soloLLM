# SoloLLM V3 Data Plan

V3 should start as a data-quality and data-diversity project. V2 is already close to GPT-2 small on the project held-out split, but it loses much more clearly on WikiText-2, LAMBADA, multiple-choice continuation scoring, and generation repetition. That points to a distribution/generalization gap more than a basic architecture failure.

## Final Dataset Status

The final v3 dataset was built successfully at:

```text
/home/bmx/_projects/soloLLM/data/v3_10b_1024
```

Final stats:

| Item | Value |
| --- | ---: |
| Accepted tokens | 10,000,000,534 |
| Train tokens | 9,800,728,576 |
| Validation tokens | 99,422,208 |
| Test tokens | 99,848,192 |
| Context length | 1024 |
| Train shards | `0:1168` |
| Validation shards | `1169:1180` |
| Test shards | `1181:1192` |

The final 123M run used one clean pass over the train split. The final 150M run
used the same 10B dataset path and became the best-performing v3 checkpoint.

Final source mix:

| Source | Accepted tokens | Share |
| --- | ---: | ---: |
| FineWeb-Edu `sample-10BT` | 4,000,001,532 | 40% |
| DCLM baseline | 2,500,001,319 | 25% |
| FineWeb `sample-10BT` | 1,499,997,774 | 15% |
| English Wikipedia | 999,998,937 | 10% |
| OpenWebText | 1,000,000,972 | 10% |

The build stats are stored in
`/home/bmx/_projects/soloLLM/data/v3_10b_1024/build_stats.json`.

## Goal

Build a clean 1024-token pilot corpus first, then run short data-ablation pilots before committing to the long v3 train.

Initial pilot target:

| Item | Target |
| --- | ---: |
| Total pilot tokens | 1,000,000,000 |
| Context length | 1024 |
| Train split | 98% |
| Validation split | 1% |
| Test split | 1% |
| Tokenizer | GPT-2 tokenizer |
| Shard format | PyTorch tensor, `(rows, 1024)` |

## Initial Source Mix

The first manifest is `sologpt_v3/data_sources.yaml`.

| Source | Weight | Reason |
| --- | ---: | --- |
| FineWeb-Edu `sample-10BT` | 40% | High-quality educational web text; strongest first bet for external generalization. |
| DCLM baseline | 25% | Curated web baseline with strong data-quality emphasis. |
| FineWeb `sample-10BT` | 15% | Broader web coverage so the model does not become too textbook-like. |
| English Wikipedia | 10% | Long-form encyclopedic prose; useful for WikiText-style gaps. |
| OpenWebText | 10% | Preserves the project distribution that v2 already handles well. |

Optional later sources:

- Dolma or Dolma 3 subsets, once the local dataset loader path is clean.
- A small licensed code/docs slice, such as The Stack v2, after access/licensing is confirmed.
- Public-domain book/prose data if licensing and contamination checks are straightforward.

## Curation Rules

The first builder performs conservative filtering:

- stream datasets instead of mirroring raw corpora,
- interleave enabled sources by token quota,
- normalize whitespace,
- reject very short and very long documents,
- reject low English/language-score documents when metadata is available,
- reject repeated-line-heavy documents,
- exact-deduplicate normalized documents by hash,
- assign train/val/test split by document hash,
- pack documents with EOS separators into 1024-token rows.

Near-deduplication and benchmark contamination filtering should be added before final 8B-12B-token v3 training. The 1B pilot is meant to validate the data mix and pipeline first.

## Pilot Evaluation

Do not judge data quality only by project validation PPL. Each 300M-500M-token data pilot should evaluate:

- project validation PPL,
- capped WikiText-2 PPL,
- capped LAMBADA PPL,
- LAMBADA last-token/last-word accuracy,
- fixed prompt generation metrics.

The winning data mix is the one that improves external validation fastest without damaging project validation.

## Commands

Smoke-test the pipeline:

```bash
python scripts/build_v3_pilot_dataset.py \
  --manifest sologpt_v3/data_sources.yaml \
  --output-dir outputs/v3_data/pilot_1b_smoke \
  --target-tokens 200000 \
  --shard-tokens 32768 \
  --shuffle-buffer 0 \
  --overwrite
```

Build the 1B-token pilot:

```bash
python scripts/build_v3_pilot_dataset.py \
  --manifest sologpt_v3/data_sources.yaml \
  --output-dir /home/bmx/_projects/soloLLM/data/v3_pilot_1b_1024 \
  --target-tokens 1000000000
```

The builder writes split-specific shards under `train/`, `val/`, and `test/`, plus hardlinked combined shards under `chunks/`. Use `splits.json` for the train/val/test shard ranges.

Build the final 10B-token dataset with notification wrapper:

```bash
bash scripts/build_v3_10b_dataset_notify.sh
```

The final combined shard ranges are recorded in
`/home/bmx/_projects/soloLLM/data/v3_10b_1024/splits.json`.
