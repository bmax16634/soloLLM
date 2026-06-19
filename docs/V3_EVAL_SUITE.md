# SoloLLM V3 Eval Suite

This document defines the executable v3 eval suite. The goal is to make the final v3 comparison repeatable against fixed GPT-2 and v2 baselines.

## Runner

Use `eval/v3_eval_suite.py`.

By default it writes a manifest only. Add `--execute` to run the commands.

```bash
python -m eval.v3_eval_suite \
  --candidate-label v3 \
  --candidate-loader v2 \
  --candidate-checkpoint outputs/sologpt_v3/final/checkpoints/latest.pt \
  --candidate-config sologpt_v3/config.json \
  --output-dir outputs/eval_suites/v3_final \
  --report-dir docs/results/v3_final_eval \
  --device cuda \
  --no-progress
```

`--candidate-loader v2` means the checkpoint uses the current `SoloGPT_v2` model class. If v3 introduces a new model class, add a loader before running the suite.

## Components

The full suite contains:

| Component | Script | Output |
| --- | --- | --- |
| Held-out project PPL | `eval/eval.py` | candidate and GPT-2 JSON |
| Fixed prompt generations | `eval/generate_samples.py` | JSON and Markdown samples |
| Generation metrics | `eval/generation_metrics.py` | JSON and Markdown metrics |
| External PPL/LAMBADA | `eval/external_benchmarks.py` | full WikiText-2 and LAMBADA report |
| Multiple-choice scoring | `eval/multiple_choice_benchmarks.py` | HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande report |

## Final V3 Runs

The final v3 suite was run for both the best 150M checkpoint and the smaller
123M ablation.

| Run | Candidate | Output directory | Status |
| --- | --- | --- | --- |
| v3 150M final | `v3_fresh_10b_150m_10bdata` | `outputs/eval_suites/v3_fresh_10b_150m_10bdata_gpt2_full_suite/` | complete |
| v3 123M final | `v3_fresh_10b_123m_10bdata` | `outputs/eval_suites/v3_fresh_10b_123m_10bdata_gpt2_full_suite/` | complete |

Headline results:

| Check | GPT-2 small | v3 123M | v3 150M | Read |
| --- | ---: | ---: | ---: | --- |
| Project held-out PPL | 25.32 | 25.64 | 24.90 | 150M wins; 123M narrowly loses |
| WikiText-2 PPL | 45.32 | 41.87 | 41.18 | both v3 models win |
| LAMBADA PPL | 40.62 | 36.28 | 35.35 | both v3 models win |
| LAMBADA last-word exact | 32.60% | 32.80% | 33.07% | both v3 models win slightly |
| Multiple-choice avg acc norm | 41.05% | 42.46% | 42.71% | both v3 models win |
| Mean repeated bigram fraction | 17.15% | 18.05% | 17.23% | GPT-2 remains slightly less repetitive |

The final report is `docs/results/v3_final_gpt2_comparison.md`. The retained
local eval outputs are also indexed in `docs/V3_ARTIFACT_MANIFEST.md`.

Run only selected components with `--components`:

```bash
python -m eval.v3_eval_suite \
  --components external multiple_choice \
  --candidate-label v3 \
  --candidate-checkpoint outputs/sologpt_v3/final/checkpoints/latest.pt \
  --candidate-config sologpt_v3/config.json \
  --execute
```

## Final V3 Claim Standard

V3 should only claim it beats GPT-2 small broadly if it beats GPT-2 small on:

- project held-out PPL,
- WikiText-2 PPL,
- LAMBADA PPL,
- LAMBADA last-token or last-word accuracy,
- most length-normalized multiple-choice benchmarks,
- generation repetition metrics without worse qualitative samples.

The current v2 5.60B checkpoint and GPT-2 small are the fixed baselines.

Final outcome: the 150M checkpoint supports the broad overall GPT-2 comparison
claim, with caveats. The 123M checkpoint supports a strong smaller-model ablation
but not the strict claim that a smaller-than-GPT-2 model beats GPT-2 across every
metric.

## Frozen V2 Baseline Run

The v2 closeout baseline was run with this suite and finished successfully on June 9, 2026 at `03:24 UTC`.

| Item | Value |
| --- | --- |
| Candidate | `v2_5p6b` |
| Baseline | `gpt2` |
| Output directory | `outputs/eval_suites/v2_5p6b_gpt2_full_suite/` |
| Report directory | `outputs/eval_suites/v2_5p6b_gpt2_full_suite/reports/` |
| Suite status | exit status `0` |

Headline results:

| Check | v2 5.60B | GPT-2 small | Result |
| --- | ---: | ---: | --- |
| Project held-out PPL | 25.56 | 25.32 | v2 `+0.95%` PPL |
| WikiText-2 PPL | 84.81 | 49.86 | GPT-2 stronger |
| LAMBADA PPL | 63.03 | 42.26 | GPT-2 stronger |
| LAMBADA last-word exact | 29.09% | 32.60% | GPT-2 stronger |
| Multiple-choice avg acc norm | 38.97% | 41.05% | GPT-2 stronger |
| Mean repeated bigram fraction | 20.15% | 17.15% | GPT-2 less repetitive |

Use this run as the frozen v2 baseline when judging final v3. V3 needs to beat GPT-2 directly, not only beat v2.

## Quick Dry Run

To verify commands without running long evals:

```bash
python -m eval.v3_eval_suite \
  --candidate-label v3-smoke \
  --output-dir outputs/eval_suites/v3_smoke \
  --report-dir docs/results/v3_smoke_eval \
  --components heldout external multiple_choice \
  --heldout-max-batches 1 \
  --max-wikitext-tokens 1000 \
  --max-lambada-examples 5 \
  --max-mc-examples 5
```

This writes `suite_manifest.json` and prints the commands. Add `--execute` only when you want to run them.
