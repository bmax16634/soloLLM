# Phase 0 Checklist

Phase 0 means the project is ready for real v2 GPU sanity runs. It does not mean the final model has been trained.

## Acceptance Criteria

| Area | Status | Evidence |
| --- | --- | --- |
| Clean model import | Done | `from sologpt_v2 import SoloGPT_v2` works. |
| v2 forward pass | Done | `tests/test_models.py` checks output shape. |
| Context validation | Done | v2 rejects sequences longer than `max_seq_len`. |
| Parameter accounting | Done | v2 reports `123,616,512` parameters for the default config. |
| Training CLI | Done | `python -m sologpt_v2.pretrain --help` works. |
| Dry-run training | Done | pytest runs tiny CPU training without OpenWebText. |
| Checkpoint resume | Done | pytest resumes from `checkpoints/latest.pt`. |
| Eval CLI | Done | `python eval/eval.py --help` works from repo root. |
| Eval smoke test | Done | pytest writes finite v2 eval JSON on a tiny shard. |
| Clean-clone tests | Done | `python -m pytest` does not require GPU, internet, OpenWebText, or model weights. |
| Public docs | Done | README has test, train, resume, and eval commands. |
| Portfolio brief | Done | `docs/PROJECT_BRIEF.md` frames v1, v2, eval, and remaining work. |

## Verification Commands

```bash
python -m pytest
git diff --check
python -m sologpt_v2.pretrain --help
python eval/eval.py --help
```

## Phase 0 Output

The default v2 model is a GPT-2-small-scale control model:

- 12 layers
- 768 embedding width
- 12 heads
- 512-token context
- RoPE attention
- pre-norm blocks
- tied embeddings
- 123,616,512 parameters

## Not Included In Phase 0

These belong to later phases:

- 50M-token GPU sanity run,
- 300M-token pilot run,
- final 3B-4B-token training run,
- final v1/v2/GPT-2 perplexity table,
- generation samples from the trained v2 checkpoint,
- Hugging Face publication.
