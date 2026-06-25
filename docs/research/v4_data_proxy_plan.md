# SoloLLM v4 Data Proxy Plan

The Phase 1 architecture proxy did not justify switching to the deeper/narrower SmolLM2-style shape by itself. The next controlled experiment is data quality and data mix.

## Question

Can a better data mix close more of the gap to SmolLM/SmolLM2 than architecture shape alone?

## Fixed Controls

- Architecture: v3-style 57M proxy from `sologpt_v3/config_proxy_v3_style_60m_1024.json`
- Training budget: 300M tokens per data mix
- Context length: 1024
- Tokenizer: GPT-2 tokenizer
- Eval target: validation loss first, then the modern eval suite for promising checkpoints

## Baseline

The v3-style architecture proxy on the existing v3 pilot dataset reached:

| Dataset | Best val loss | Best val PPL |
|---|---:|---:|
| v3 pilot mix | 4.1530 | 63.63 |

## Runnable Mixes

### `quality_web`

Manifest: `sologpt_v4/data_mix_quality_web_300m.yaml`

Purpose: test whether a cleaner web-heavy mix improves general validation without adding much synthetic/math specialization.

| Source | Weight | Why |
|---|---:|---|
| SmolLM FineWeb-Edu dedup | 45% | high-quality deduplicated educational web |
| DCLM baseline | 25% | curated web data-quality baseline |
| FineWeb sample-10BT | 15% | broad web coverage |
| Wikipedia | 10% | factual/reference prose |
| OpenWebText | 5% | legacy project distribution |

### `smollm_reasoning`

Manifest: `sologpt_v4/data_mix_smollm_reasoning_300m.yaml`

Purpose: test a more SmolLM-inspired mix with educational web, synthetic textbook-style content, and math reasoning.

| Source | Weight | Why |
|---|---:|---|
| SmolLM FineWeb-Edu dedup | 35% | high-quality educational web |
| Cosmopedia v2 | 25% | synthetic textbook/blog/story content |
| FineMath-4+ | 15% | math reasoning and worked explanations |
| DCLM baseline | 15% | curated web diversity |
| Wikipedia | 5% | reference prose |
| OpenWebText | 5% | legacy project distribution |

## New Dataset Candidates

Use now:

- `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup`
- `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2`
- `HuggingFaceTB/finemath`, `finemath-4plus`

Keep optional:

- `HuggingFaceTB/stack-edu`
- `HuggingFaceTB/smollm-corpus`, `python-edu`

The optional code sources currently expose Software Heritage blob IDs rather than direct file text in the dataset rows. They should not be enabled until the builder has a resolver for Software Heritage content and the licensing policy is documented.

Do not prioritize now:

- Dolma. It is valuable and broad, but the current `datasets` path for `allenai/dolma` requires a deprecated dataset script path in this environment. The first v4 data proxy should use sources that stream directly.

## Decision Rule

After each 300M-token proxy train, compare against the v3 pilot baseline:

- primary: best validation loss/PPL
- secondary: WikiText BPB, LAMBADA, HellaSwag, ARC

If a mix beats the v3 pilot baseline on validation, run the modern eval suite on that checkpoint. If neither mix beats the baseline, improve filtering/dedup before adding more sources.
