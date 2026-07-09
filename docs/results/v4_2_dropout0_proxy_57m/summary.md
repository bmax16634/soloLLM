# V4.2 Dropout-0 Proxy Summary

Controlled 57M/300M-token proxy using the v4.1b data mix. The only intended model/training-recipe change is dropout `0.1 -> 0.0`.

| Proxy | Val PPL ↓ | Train loss ↓ | WikiText PPL ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| v4.1b dropout0.1 | 54.46 | 4.1755 | 149.37 | 68.55 | 0.2381 | 0.3613 |
| v4.2 dropout0.0 | 51.28 | 4.0330 | 149.84 | 65.44 | 0.2366 | 0.3678 |

## Multiple Choice

| Proxy | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande |
|---|---:|---:|---:|---:|---:|
| v4.1b dropout0.1 | 0.2504 | 0.5570 | 0.2965 | 0.2007 | 0.5020 |
| v4.2 dropout0.0 | 0.2566 | 0.5800 | 0.3070 | 0.2007 | 0.4949 |

## Decision

Dropout `0.0` is a promising recipe change. It improves proxy validation,
LAMBADA perplexity, and MC average versus the v4.1b dropout-0.1 baseline.

The tradeoff is not perfect: WikiText PPL is basically flat/slightly worse
(`149.84` vs `149.37`), LAMBADA token accuracy is slightly lower, and
WinoGrande regresses. But the overall signal is better than the data-only
interpolation experiments.

Next: test dropout `0.0` on the broader v4.1 data mix before committing to a
full 150M/10B train. That checks whether the recipe improvement transfers to
the best broad-external mix.
