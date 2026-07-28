# Hugging Face Publishing Workflow

This folder tracks the Hugging Face publishing manifest for SoloLLM.

Development stays in `/home/bmx/_projects/soloLLM`. The Hugging Face repos are publishing targets, not the normal work area.

## Layout

- `repos.yaml` maps SoloLLM artifacts to Hugging Face repos, local publishing clones, generated export folders, and planned fileserver mirrors.
- Generated HF packages live under `outputs/huggingface/` and are ignored by the SoloLLM Git repo.
- Real local Hugging Face clones live under `/home/bmx/_hf_repos/`.
- Fileserver mirrors/backups should live under `/mnt/raid/hf_repos/` once created and verified.

## Agent Rules

1. Work in the SoloLLM project root for source, training, eval, docs, and export generation.
2. Use `repos.yaml` to find the export folder and matching HF clone.
3. Dry-run sync before applying changes to an HF clone.
4. Run `git status --short --branch` in the HF clone after sync.
5. Run `git lfs status` before publishing model repos.
6. Commit/push only with explicit approval.
7. If a README/model card is edited directly in an HF clone, copy the durable change back into tracked SoloLLM files before the next export.

General Ariya convention: `[[hugging-face-publishing-workflow]]`.
