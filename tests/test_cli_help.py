import subprocess
import sys
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_pretrain_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "-m", "solollm.train", "--help"],
    )

    assert "--max-eval-tokens" in result.stdout


def test_eval_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "eval/eval.py", "--help"],
    )

    assert "--model" in result.stdout


def test_generate_samples_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "-m", "eval.generate_samples", "--help"],
    )

    assert "--max-new-tokens" in result.stdout


def test_generation_metrics_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "-m", "eval.generation_metrics", "--help"],
    )

    assert "--input-json" in result.stdout


def test_external_benchmarks_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "-m", "eval.external_benchmarks", "--help"],
    )

    assert "--benchmarks" in result.stdout


def test_multiple_choice_benchmarks_help_runs_from_repo_root():
    result = run_command(
        [sys.executable, "-m", "eval.multiple_choice_benchmarks", "--help"],
    )

    assert "--benchmarks" in result.stdout


def test_v3_eval_suite_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "eval.v3_eval_suite", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--execute" in result.stdout
