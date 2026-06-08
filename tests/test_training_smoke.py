import json
import os
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "v2_tiny_config.json"
MODERN_FIXTURE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "v2_tiny_modern_config.json"


def run_command(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def make_tiny_shard(shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    torch.save(torch.randint(0, 64, (8, 16), dtype=torch.long), shard_dir / "shard_00000.pt")


def test_v2_training_dry_run_writes_artifacts_and_resumes(tmp_path):
    shard_dir = tmp_path / "shards"
    run_dir = tmp_path / "run"
    resumed_dir = tmp_path / "resumed"
    make_tiny_shard(shard_dir)

    run_command(
        [
            sys.executable,
            "-m",
            "sologpt_v2.pretrain",
            "--config",
            str(FIXTURE_CONFIG),
            "--shard-dir",
            str(shard_dir),
            "--output-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--dry-run",
            "--max-steps",
            "2",
            "--no-progress",
        ]
    )

    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "metrics_summary.json"
    latest_path = run_dir / "checkpoints" / "latest.pt"

    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert latest_path.is_file()
    assert (run_dir / "final_model.pt").is_file()

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["status"] == "complete"
    assert summary["optimizer_steps"] >= 2
    assert summary["final_train_loss"] > 0

    run_command(
        [
            sys.executable,
            "-m",
            "sologpt_v2.pretrain",
            "--config",
            str(FIXTURE_CONFIG),
            "--shard-dir",
            str(shard_dir),
            "--output-dir",
            str(resumed_dir),
            "--resume",
            str(latest_path),
            "--device",
            "cpu",
            "--dry-run",
            "--max-steps",
            "3",
            "--no-progress",
        ]
    )

    with open(resumed_dir / "metrics_summary.json", "r", encoding="utf-8") as f:
        resumed_summary = json.load(f)
    assert resumed_summary["status"] == "complete"
    assert resumed_summary["optimizer_steps"] >= 3


def test_v2_modern_training_dry_run_writes_artifacts(tmp_path):
    shard_dir = tmp_path / "shards"
    run_dir = tmp_path / "modern_run"
    make_tiny_shard(shard_dir)

    run_command(
        [
            sys.executable,
            "-m",
            "sologpt_v2.pretrain",
            "--config",
            str(MODERN_FIXTURE_CONFIG),
            "--shard-dir",
            str(shard_dir),
            "--output-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--dry-run",
            "--max-steps",
            "1",
            "--no-progress",
        ]
    )

    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "metrics_summary.json").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
