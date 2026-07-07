# V4.1b Blend Proxy Modern Eval Summary

The 57M `v4.1b_blend` proxy rebalances v4.1 by adding more SmolLM/Cosmopedia/FineMath signal while keeping most of the broad-web recovery.

| Proxy | WikiText BPB ↓ | WikiText PPL ↓ | LAMBADA BPB ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| smollm_reasoning | 1.6170 | 155.71 | 1.5919 | 72.56 | 0.2271 | 0.3686 |
| v4.1_blend | 1.5937 | 144.80 | 1.5619 | 66.93 | 0.2408 | 0.3601 |
| v4.1b_blend | 1.6037 | 149.37 | 1.5708 | 68.55 | 0.2381 | 0.3613 |

## Multiple Choice

| Proxy | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande |
|---|---:|---:|---:|---:|---:|
| smollm_reasoning | 0.2514 | 0.5680 | 0.3123 | 0.1973 | 0.5138 |
| v4.1_blend | 0.2476 | 0.5610 | 0.2860 | 0.2040 | 0.5020 |
| v4.1b_blend | 0.2504 | 0.5570 | 0.2965 | 0.2007 | 0.5020 |

## Decision

Do not scale the exact v4.1b mix. It improves MC average only slightly versus
v4.1 (`0.3613` vs `0.3601`) while worsening WikiText and LAMBADA. It also does
not recover the reasoning-heavy proxy's MC strength.

This closes the current data-mix proxy loop: the project has a measurable
broad-LM versus reasoning/MC tradeoff, but no blended proxy has earned a full
150M/10B scale-up yet.
