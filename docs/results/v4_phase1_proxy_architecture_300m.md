# V4 Phase 1 Architecture Proxy Results

Matched-parameter proxy comparison for the SoloLLM-Modern architecture hypothesis.

- Target tokens per proxy: `300,000,000`
- Dataset: `v3_pilot_1b_1024`
- Purpose: compare wider/shallower v3-style shape against deeper/narrower SmolLM2-style shape before a full model train.

| Proxy | Params | Tokens | Best val loss | Best val PPL | Final train loss | Tok/s | Hours | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3-style proxy | 57117696 | 300011520 | 4.1530 | 63.63 | 4.0549 | 72164.1 | 1.15 | 12.3 |
| SmolLM2-style proxy | 57066240 | 300011520 | 4.1723 | 64.86 | 4.0429 | 53923.4 | 1.55 | 15.3 |

Lower validation loss/PPL is better. This proxy is a first architecture signal, not the final SoloLLM-Modern result.
