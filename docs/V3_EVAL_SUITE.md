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
