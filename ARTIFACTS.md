# SoloLLM Artifacts

SoloLLM keeps public source code in git. Internal docs and generated experiment output are stored as fileserver artifacts so they do not get pushed to GitHub.

## Fileserver artifact root

```text
ben@192.168.0.193:/mnt/raid/26/aimlcs/project-artifacts/soloLLM/
```

## Artifact groups

| Artifact | Remote path | Local restore path | Notes |
|---|---|---|---|
| `docs` | `/mnt/raid/26/aimlcs/project-artifacts/soloLLM/docs/` | `docs/` | Internal research docs, result writeups, generated docs, and planning notes. |
| `outputs` | `/mnt/raid/26/aimlcs/project-artifacts/soloLLM/outputs/` | `outputs/` | Small non-checkpoint outputs: eval summaries, configs, logs, reports, and export metadata. |

Published model checkpoints are separate. See `CHECKPOINTS.md` for Hugging Face model restore instructions.

## Restore artifacts

```bash
bash scripts/restore_artifacts_from_fs.sh
```

## Sync artifacts to fileserver

```bash
bash scripts/sync_artifacts_to_fs.sh
```
