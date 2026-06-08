import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pretrain_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "sologpt_v2.pretrain", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-eval-tokens" in result.stdout


def test_eval_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "eval/eval.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--model" in result.stdout


def test_generate_samples_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "eval.generate_samples", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-new-tokens" in result.stdout


def test_generation_metrics_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "eval.generation_metrics", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--input-json" in result.stdout


def test_external_benchmarks_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "eval.external_benchmarks", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--benchmarks" in result.stdout


def test_multiple_choice_benchmarks_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "eval.multiple_choice_benchmarks", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
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
