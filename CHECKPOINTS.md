# SoloLLM Checkpoints

SoloLLM source code stays in this repository. Large model checkpoint files do not.

The durable published model checkpoints live in Hugging Face model repos, with fileserver bare mirrors as the homelab backup. Local checkpoint files under `outputs/` are disposable working artifacts unless a new checkpoint is explicitly promoted into this manifest.

## Published model checkpoints

| Model | Hugging Face repo | Local publishing clone | Fileserver mirror | Durable files |
|---|---|---|---|---|
| `solollm-1.0-123m-base` | `https://huggingface.co/bmax16634/solollm-1.0-123m-base` | `/home/bmx/_hf_repos/solollm/models/solollm-1.0-123m-base` | `ben@192.168.0.193:/mnt/raid/26/aimlcs/hf_repos/solollm/models/solollm-1.0-123m-base.git` | `model.safetensors`, `original/final_model.pt` |
| `solollm-1.0-152m-base` | `https://huggingface.co/bmax16634/solollm-1.0-152m-base` | `/home/bmx/_hf_repos/solollm/models/solollm-1.0-152m-base` | `ben@192.168.0.193:/mnt/raid/26/aimlcs/hf_repos/solollm/models/solollm-1.0-152m-base.git` | `model.safetensors`, `original/final_model.pt` |

## Restore published checkpoints

From the SoloLLM project root:

```bash
bash scripts/restore_published_models_from_hf.sh
```

That script updates or clones the publishing repos under `/home/bmx/_hf_repos/solollm/models/`, runs `git lfs pull`, and refreshes disposable local copies under `outputs/huggingface/`.

For fileserver-first restore instead of Hugging Face:

```bash
SOLOLLM_MODEL_SOURCE=fs bash scripts/restore_published_models_from_hf.sh
```

## Cleanup policy

Safe to delete locally after the repos above are synced:

- `outputs/**/*.pt`
- `outputs/**/*.pth`
- `outputs/**/*.bin`
- `outputs/**/*.safetensors`

Do not delete `data/` unless it has its own dataset artifact/restore manifest.

Keep small eval summaries, configs, READMEs, and leaderboard/report files when useful because they are cheap and preserve experiment context.
