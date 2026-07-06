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

## Completed Proxy Results

| Data mix | Best val loss | Best val PPL | Decision |
|---|---:|---:|---|
| v3 pilot mix | 4.1530 | 63.63 | baseline |
| quality_web | 4.1926 | 66.20 | reject |
| smollm_reasoning | 3.8521 | 47.09 | scale to full run |

The `smollm_reasoning` proxy was a large validation win, so it was scaled to a
full 151.9M-parameter 10B-token run.

## Full-Scale V4 Result

The scaled run answered the real question: the proxy result did not transfer
cleanly to the final general-LM objective.

| Benchmark | GPT-2 small | v3 150M | v4 smollm_reasoning 150M |
|---|---:|---:|---:|
| Project held-out PPL | 25.32 | 24.90 | 27.38 |
| WikiText-2 PPL | 45.32 | 41.18 | 43.59 |
| LAMBADA PPL | 40.62 | 35.35 | 38.46 |
| MC avg acc norm | 41.05% | 42.71% | 43.52% |
| Repeated bigram fraction | 17.15% | 17.23% | 8.55% |

Read:

- The reasoning-heavy mix improved multiple-choice scoring and repetition
  diagnostics.
- It hurt broad general-LM perplexity relative to v3.
- V3 remains the best general SoloLLM base model.

Full report: `docs/results/v4_smollm_reasoning_150m_10b.md`.

## V4.1 Follow-Up

Do not keep training the completed v4 checkpoint as the main model. The next
experiment should be `v4.1`: a blended data-mix ablation that keeps the useful
reasoning signal but restores broad general text.

Suggested v4.1 mix:

- 60-75% v3-style broad web/reference data,
- 15-25% SmolLM reasoning/educational/math data,
- 5-10% Wikipedia/reference/code-light high-signal data.

Run a 57M proxy before another 150M 10B train. Scale only if the proxy improves
both validation PPL and modern-suite benchmark scores.

Concrete proxy candidate:

Manifest: `sologpt_v4/data_mix_v4_1_blend_300m.yaml`

Launcher: `scripts/run_v4_1_blend_proxy_notify.sh`

| Source group | Weight | Purpose |
|---|---:|---|
| v3-style broad web/reference | 72% | recover broad general-LM perplexity |
| SmolLM-style synthetic/math/reasoning | 20% | keep the v4 reasoning and repetition gains |
| Wikipedia/reference | 8% | preserve encyclopedic/reference style |

This should be judged against both earlier proxies:

| Data mix | Best val loss | Best val PPL | Read |
|---|---:|---:|---|
| v3 pilot mix | 4.1530 | 63.63 | broad baseline |
| quality_web | 4.1926 | 66.20 | cleaner web alone did not transfer |
| smollm_reasoning | 3.8521 | 47.09 | proxy win, but full-scale broad PPL regression |
| v4.1 blend | 4.0682 | 58.45 | better than v3 pilot, weaker than reasoning-heavy proxy |

Scale v4.1 only if it preserves most of the proxy validation gain without
showing the broad-PPL regression pattern from the full v4 run.

The first v4.1 proxy completed successfully. It supports the blended-data idea
more than the rejected `quality_web` mix, but not strongly enough to justify an
immediate full 150M train. Run the modern eval suite on the v4.1 proxy
checkpoint first.

The modern eval confirmed the tradeoff:

| Proxy | WikiText PPL | LAMBADA PPL | LAMBADA tok acc | MC avg norm |
|---|---:|---:|---:|---:|
| smollm_reasoning | 155.71 | 72.56 | 0.2271 | 0.3686 |
| v4.1 blend | 144.80 | 66.93 | 0.2408 | 0.3601 |

This is better broad-LM transfer, but worse MC transfer. The exact v4.1 blend
should not be scaled directly; use it to design a second blend with slightly
more reasoning/math signal.

## V4.1b Follow-Up

The next proxy should test the midpoint between the reasoning-heavy run and the
v4.1 broad blend.

Manifest: `sologpt_v4/data_mix_v4_1b_blend_300m.yaml`

Launcher: `scripts/run_v4_1b_blend_proxy_eval_notify.sh`

| Source group | Weight | Purpose |
|---|---:|---|
| v3-style broad web | 60% | keep the v4.1 WikiText/LAMBADA recovery |
| SmolLM/Cosmopedia/FineMath reasoning | 34% | restore some MC strength |
| Wikipedia/reference | 6% | preserve reference prose |

Decision rule:

- Scale only if v4.1b improves MC versus v4.1 without giving back most of the
  WikiText/LAMBADA improvement.
- If v4.1b still fails the tradeoff, stop data-mix proxy search and move to a
  separate architecture/tokenizer experiment instead of running another 150M
  10B train.
