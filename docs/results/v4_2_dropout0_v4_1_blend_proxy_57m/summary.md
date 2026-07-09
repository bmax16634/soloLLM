# V4.2 Dropout-0 On V4.1 Blend Proxy Summary

Controlled 57M/300M-token proxy using the broader v4.1 blend data. The intended change versus v4.1 is dropout `0.1 -> 0.0`.

| Proxy | Val PPL ↓ | Train loss ↓ | WikiText PPL ↓ | LAMBADA PPL ↓ | LAMBADA tok acc ↑ | MC avg norm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| v4.1 blend dropout0.1 | 58.45 | 4.0757 | 144.80 | 66.93 | 0.2408 | 0.3601 |
| v4.1b blend dropout0.1 | 54.46 | 4.1755 | 149.37 | 68.55 | 0.2381 | 0.3613 |
| v4.1b blend dropout0.0 | 51.28 | 4.0330 | 149.84 | 65.44 | 0.2366 | 0.3678 |
| v4.1 blend dropout0.0 | 54.98 | 3.9285 | 141.55 | 64.54 | 0.2420 | 0.3705 |

## Multiple Choice

| Proxy | HellaSwag | PIQA | ARC-Easy | ARC-Challenge | WinoGrande |
|---|---:|---:|---:|---:|---:|
| v4.1 blend dropout0.1 | 0.2476 | 0.5610 | 0.2860 | 0.2040 | 0.5020 |
| v4.1b blend dropout0.1 | 0.2504 | 0.5570 | 0.2965 | 0.2007 | 0.5020 |
| v4.1b blend dropout0.0 | 0.2566 | 0.5800 | 0.3070 | 0.2007 | 0.4949 |
| v4.1 blend dropout0.0 | 0.2505 | 0.5780 | 0.3105 | 0.2107 | 0.5028 |

## Decision

This passes the scale rule. Compared with v4.1 dropout-0.1, the dropout-0.0
run improves validation, WikiText, LAMBADA, LAMBADA token accuracy, and MC
average. It is also the strongest proxy so far on the combined external suite.

Next: build a full-size 10B-token v4.2 dataset with the v4.1 blend recipe and
train a fresh 150M model with dropout `0.0`. Do not train the full model by
repeating this 300M proxy dataset.
