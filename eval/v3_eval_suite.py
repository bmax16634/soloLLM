"""Orchestrate the full v3 evaluation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "sologpt_v2"
    / "stretch_5p85b_modern_small_from3b"
    / "checkpoints"
    / "latest.pt"
)
DEFAULT_CONFIG = REPO_ROOT / "sologpt_v2" / "config_modern_small.json"
EXTERNAL_SHARD_DIR = REPO_ROOT.parents[3] / "soloLLM" / "data" / "tokenized_chunks"
DEFAULT_SHARD_DIR = EXTERNAL_SHARD_DIR if EXTERNAL_SHARD_DIR.exists() else REPO_ROOT / "data" / "tokenized_chunks"


COMPONENTS = ["heldout", "generation", "generation_metrics", "external", "multiple_choice"]


@dataclass
class SuiteCommand:
    name: str
    component: str
    command: list[str]
    outputs: list[Path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the full SoloLLM v3 eval suite")
    parser.add_argument("--candidate-label", default="v3", help="Display label for the candidate model")
    parser.add_argument("--candidate-loader", default="v2", choices=["v2"], help="Underlying local loader for the candidate")
    parser.add_argument("--candidate-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--candidate-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--baseline-label", default="gpt2")
    parser.add_argument("--output-dir", default="outputs/eval_suites/v3_eval")
    parser.add_argument("--report-dir", default="docs/results/v3_eval_suite")
    parser.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    parser.add_argument("--shards", default="58:60")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heldout-max-batches", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-wikitext-tokens", type=int, default=0, help="0 means full WikiText-2 test")
    parser.add_argument("--max-lambada-examples", type=int, default=0, help="0 means full LAMBADA test")
    parser.add_argument("--max-mc-examples", type=int, default=0, help="0 means full validation splits")
    parser.add_argument("--components", nargs="+", default=["all"], choices=["all", *COMPONENTS])
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Without this, only writes a manifest.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def selected_components(values: list[str]) -> set[str]:
    return set(COMPONENTS) if "all" in values else set(values)


def maybe_device_args(device: str | None) -> list[str]:
    return ["--device", device] if device else []


def maybe_no_progress(flag: bool) -> list[str]:
    return ["--no-progress"] if flag else []


def build_commands(args: argparse.Namespace) -> list[SuiteCommand]:
    python = sys.executable
    output_dir = resolve_path(args.output_dir)
    report_dir = resolve_path(args.report_dir)
    checkpoint = resolve_path(args.candidate_checkpoint)
    config = resolve_path(args.candidate_config)
    shard_dir = resolve_path(args.shard_dir)
    chosen = selected_components(args.components)
    commands: list[SuiteCommand] = []

    candidate_heldout = output_dir / f"{args.candidate_label}_heldout.json"
    baseline_heldout = output_dir / f"{args.baseline_label}_heldout.json"
    if "heldout" in chosen:
        candidate_cmd = [
            python,
            "eval/eval.py",
            "--model",
            args.candidate_loader,
            "--model-label",
            args.candidate_label,
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--shard-dir",
            str(shard_dir),
            "--shards",
            args.shards,
            "--batch-size",
            str(args.batch_size),
            "--output-json",
            str(candidate_heldout),
            *maybe_device_args(args.device),
            *maybe_no_progress(args.no_progress),
        ]
        if args.heldout_max_batches is not None:
            candidate_cmd.extend(["--max-batches", str(args.heldout_max_batches)])
        commands.append(SuiteCommand("candidate_heldout", "heldout", candidate_cmd, [candidate_heldout]))

        baseline_cmd = [
            python,
            "eval/eval.py",
            "--model",
            "gpt2",
            "--model-label",
            args.baseline_label,
            "--shard-dir",
            str(shard_dir),
            "--shards",
            args.shards,
            "--batch-size",
            str(args.batch_size),
            "--output-json",
            str(baseline_heldout),
            *maybe_device_args(args.device),
            *maybe_no_progress(args.no_progress),
        ]
        if args.heldout_max_batches is not None:
            baseline_cmd.extend(["--max-batches", str(args.heldout_max_batches)])
        commands.append(SuiteCommand("baseline_heldout", "heldout", baseline_cmd, [baseline_heldout]))

    generations_json = output_dir / "generations_candidate_gpt2.json"
    generations_md = report_dir / "generations_candidate_gpt2.md"
    if "generation" in chosen:
        commands.append(
            SuiteCommand(
                "generations",
                "generation",
                [
                    python,
                    "-m",
                    "eval.generate_samples",
                    "--models",
                    args.candidate_loader,
                    "gpt2",
                    "--v2-label",
                    args.candidate_label,
                    "--v2-checkpoint",
                    str(checkpoint),
                    "--v2-config",
                    str(config),
                    "--output-json",
                    str(generations_json),
                    "--output-md",
                    str(generations_md),
                    *maybe_device_args(args.device),
                ],
                [generations_json, generations_md],
            )
        )

    generation_metrics_json = output_dir / "generation_metrics_candidate_gpt2.json"
    generation_metrics_md = report_dir / "generation_metrics_candidate_gpt2.md"
    if "generation_metrics" in chosen:
        commands.append(
            SuiteCommand(
                "generation_metrics",
                "generation_metrics",
                [
                    python,
                    "-m",
                    "eval.generation_metrics",
                    "--input-json",
                    str(generations_json),
                    "--output-json",
                    str(generation_metrics_json),
                    "--output-md",
                    str(generation_metrics_md),
                ],
                [generation_metrics_json, generation_metrics_md],
            )
        )

    external_json = output_dir / "external_full_candidate_gpt2.json"
    external_md = report_dir / "external_full_candidate_gpt2.md"
    if "external" in chosen:
        commands.append(
            SuiteCommand(
                "external_full",
                "external",
                [
                    python,
                    "-m",
                    "eval.external_benchmarks",
                    "--models",
                    args.candidate_loader,
                    "gpt2",
                    "--v2-label",
                    args.candidate_label,
                    "--v2-checkpoint",
                    str(checkpoint),
                    "--v2-config",
                    str(config),
                    "--context-length",
                    str(args.context_length),
                    "--max-wikitext-tokens",
                    str(args.max_wikitext_tokens),
                    "--max-lambada-examples",
                    str(args.max_lambada_examples),
                    "--output-json",
                    str(external_json),
                    "--output-md",
                    str(external_md),
                    *maybe_device_args(args.device),
                    *maybe_no_progress(args.no_progress),
                ],
                [external_json, external_md],
            )
        )

    mc_json = output_dir / "multiple_choice_full_candidate_gpt2.json"
    mc_md = report_dir / "multiple_choice_full_candidate_gpt2.md"
    if "multiple_choice" in chosen:
        commands.append(
            SuiteCommand(
                "multiple_choice_full",
                "multiple_choice",
                [
                    python,
                    "-m",
                    "eval.multiple_choice_benchmarks",
                    "--models",
                    args.candidate_loader,
                    "gpt2",
                    "--v2-label",
                    args.candidate_label,
                    "--v2-checkpoint",
                    str(checkpoint),
                    "--v2-config",
                    str(config),
                    "--context-length",
                    str(args.context_length),
                    "--max-examples",
                    str(args.max_mc_examples),
                    "--output-json",
                    str(mc_json),
                    "--output-md",
                    str(mc_md),
                    *maybe_device_args(args.device),
                    *maybe_no_progress(args.no_progress),
                ],
                [mc_json, mc_md],
            )
        )

    return commands


def command_record(command: SuiteCommand) -> dict[str, Any]:
    return {
        "name": command.name,
        "component": command.component,
        "command": command.command,
        "outputs": [str(path) for path in command.outputs],
    }


def write_manifest(path: Path, args: argparse.Namespace, commands: list[SuiteCommand]) -> None:
    manifest = {
        "candidate_label": args.candidate_label,
        "candidate_loader": args.candidate_loader,
        "baseline_label": args.baseline_label,
        "components": sorted(selected_components(args.components)),
        "execute": args.execute,
        "commands": [command_record(command) for command in commands],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_command(command: SuiteCommand, *, skip_existing: bool) -> None:
    if skip_existing and all(path.exists() for path in command.outputs):
        print(json.dumps({"skipped": command.name, "outputs": [str(path) for path in command.outputs]}))
        return
    for output in command.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"running": command.name, "command": command.command}))
    subprocess.run(command.command, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolve_path(args.report_dir).mkdir(parents=True, exist_ok=True)

    commands = build_commands(args)
    manifest_path = output_dir / "suite_manifest.json"
    write_manifest(manifest_path, args, commands)
    print(json.dumps({"manifest": str(manifest_path), "commands": len(commands), "execute": args.execute}, indent=2))

    if args.execute:
        for command in commands:
            run_command(command, skip_existing=args.skip_existing)
    else:
        for command in commands:
            print(" ".join(command.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
