# V2 5.60B vs GPT-2 Full Diagnostic

This report consolidates the final v2 diagnostic comparison against GPT-2 small. It uses the 91.65M-param `v2-modern-small` 5.60B checkpoint and GPT-2 small as fixed baselines.

## Bottom Line

V2 is close to GPT-2 on the project held-out OpenWebText-style split, but GPT-2 is more robust across external text distributions and multiple-choice continuation tasks.

The largest v2 weakness is **external perplexity/generalization**, not basic next-token plausibility. LAMBADA and multiple-choice accuracy gaps are smaller than the WikiText/LAMBADA perplexity gaps.

## Core Perplexity

| Benchmark | Scope | V2 5.60B | GPT-2 small | Gap |
| --- | --- | ---: | ---: | ---: |
| Project held-out PPL | shards `58:60`, 331,353,862 tokens | 25.56 | 25.32 | v2 `+0.95%` |
| WikiText-2 PPL | full test, 285,618 scored tokens | 84.81 | 49.86 | v2 `+70.1%` |
| LAMBADA PPL | full test, 434,531 scored tokens | 63.03 | 42.26 | v2 `+49.2%` |

## LAMBADA Continuation Accuracy

| Benchmark | Scope | V2 5.60B | GPT-2 small | Gap |
| --- | --- | ---: | ---: | ---: |
| LAMBADA last-token accuracy | 5,153 examples | 44.30% | 46.67% | v2 `-2.37` points |
| LAMBADA last-word greedy exact | 5,153 examples | 29.09% | 32.60% | v2 `-3.51` points |

## Multiple-Choice Base-LM Scoring

Choices are scored by conditional log-likelihood. `Accuracy norm` is the primary metric because it normalizes by continuation length.

| Benchmark | Examples | V2 acc norm | GPT-2 acc norm | Gap |
| --- | ---: | ---: | ---: | ---: |
| HellaSwag | 10,042 | 26.99% | 29.53% | v2 `-2.54` points |
| PIQA | 1,000 | 61.00% | 63.60% | v2 `-2.60` points |
| ARC-Easy | 570 | 37.89% | 40.35% | v2 `-2.46` points |
| ARC-Challenge | 299 | 19.73% | 22.07% | v2 `-2.34` points |
| WinoGrande | 1,267 | 49.25% | 49.72% | v2 `-0.47` points |

## Generation Metrics

| Metric | V2 5.60B | GPT-2 small | Read |
| --- | ---: | ---: | --- |
| Fixed prompt corpus distinct-2 | 0.7512 | 0.7836 | GPT-2 slightly higher diversity |
| Mean repeated bigram fraction | 0.2015 | 0.1715 | GPT-2 slightly less repetitive |
| Mean repeated trigram fraction | 0.1282 | 0.1120 | GPT-2 slightly less repetitive |
| Bad-loop detections | 2 / 8 | 2 / 8 | tied on this small suite |

## Where V2 Lacks Most

Ranked by practical importance:

1. **External perplexity:** WikiText-2 and LAMBADA PPL gaps are large. This points to data distribution, data quality, and generalization as the main v3 targets.
2. **Continuation calibration:** LAMBADA token/word accuracy is close, but v2 assigns weaker probabilities overall. It often has plausible top choices, but GPT-2 is more confident and better calibrated.
3. **Multiple-choice robustness:** GPT-2 leads every length-normalized multiple-choice benchmark, but mostly by 0.5-2.6 points. This is a real gap, not an impossible one.
4. **Generation repetition:** v2 is slightly more repetitive under the same sampling settings. This should be monitored, but it is not the biggest gap.

## V3 Implications

V3 should not be only "continue v2 longer." The gaps suggest:

- improve and broaden the data mix,
- use external validation during training,
- move the main context length to 1024,
- consider GPT-2-small parity scale or a modestly larger modern config,
- keep v2 5.60B and GPT-2 small as fixed baselines,
- require v3 to beat GPT-2 on project PPL, external PPL, LAMBADA accuracy, most multiple-choice benchmarks, and generation stability before claiming it wins across the board.

## Artifacts

| Artifact | Path |
| --- | --- |
| Full external benchmark JSON | `outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/external_benchmarks_5p6b_v2_gpt2_full.json` |
| Full external benchmark report | `docs/results/external_benchmarks_5p6b_v2_gpt2_full.md` |
| Full multiple-choice JSON | `outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/multiple_choice_5p6b_v2_gpt2_full.json` |
| Full multiple-choice report | `docs/results/multiple_choice_5p6b_v2_gpt2_full.md` |
| Fixed generation metrics JSON | `outputs/sologpt_v2/stretch_5p85b_modern_small_from3b/phase4_generation_metrics_5p6b_v2_gpt2.json` |
| Fixed generation metrics report | `docs/results/phase4_generation_metrics_5p6b_v2_gpt2.md` |
