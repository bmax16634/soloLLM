#!/usr/bin/env python3
"""Build the public results page from structured leaderboard data."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "leaderboard"
OUT = ROOT / "docs" / "RESULTS.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt_params(value: Any) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.2f}M"


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value) * 100:.2f}%"


def fmt_metric(value: Any, kind: str) -> str:
    if kind == "percent":
        return fmt_percent(value)
    if kind == "integer":
        return "-" if value is None else str(int(value))
    if kind == "gb":
        return "-" if value is None else f"{float(value):.2f}GB"
    return fmt_number(value, 4 if kind == "bpb" else 2)


def get_run(by_suite: dict[str, dict[str, dict[str, Any]]], suite: str, model_id: str) -> dict[str, Any]:
    return by_suite.get(suite, {}).get(model_id, {})


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build() -> str:
    models = load_json(LEADERBOARD / "models.json")
    schema = load_json(LEADERBOARD / "metrics_schema.json")
    runs = load_jsonl(LEADERBOARD / "eval_runs.jsonl")

    by_suite: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        by_suite[row["suite"]][row["model_id"]] = row

    out: list[str] = [
        "# Results",
        "",
        "This page is generated from structured leaderboard data in `leaderboard/`.",
        "Evaluation details are described in [docs/EVALUATION_SUITE.md](EVALUATION_SUITE.md).",
        "",
        "## Current Read",
        "",
        "- SoloLLM v4.2 150M is the strongest completed SoloLLM checkpoint.",
        "- It beats GPT-2 small across the fixed project suite, but it is larger than GPT-2 small.",
        "- SoloLLM v4.2 123M is slightly smaller than GPT-2 small and wins the fixed normalized multiple-choice suite, but not every metric.",
        "- V5 moves the project toward modern small-model baselines such as SmolLM2 rather than repeatedly chasing GPT-2.",
        "",
    ]

    base_ids = [
        "gpt2_small",
        "solollm_v2_5p6b",
        "solollm_v3_123m",
        "solollm_v3_150m",
        "solollm_v4_reasoning_150m",
        "solollm_v4_2_123m",
        "solollm_v4_2_150m",
    ]
    out.extend(["## Base Core Leaderboard", ""])
    out.extend(table(
        ["Model", "Params", "Tokens", "Held-out PPL", "WikiText-2 PPL", "LAMBADA PPL", "LAMBADA tok", "LAMBADA word", "MC avg norm", "Read"],
        [[
            models[mid]["name"],
            fmt_params(models[mid]["params"]),
            models[mid]["train_tokens_display"],
            fmt_number(get_run(by_suite, "base_core_v1", mid).get("heldout_ppl")),
            fmt_number(get_run(by_suite, "base_core_v1", mid).get("wikitext_ppl")),
            fmt_number(get_run(by_suite, "base_core_v1", mid).get("lambada_ppl")),
            fmt_percent(get_run(by_suite, "base_core_v1", mid).get("lambada_tok_acc")),
            fmt_percent(get_run(by_suite, "base_core_v1", mid).get("lambada_word_acc")),
            fmt_percent(get_run(by_suite, "base_core_v1", mid).get("mc_avg_norm")),
            models[mid]["notes"],
        ] for mid in base_ids],
    ))
    out.append("")

    modern_ids = [
        "solollm_v4_2_150m",
        "solollm_v4_2_123m",
        "solollm_v3_150m",
        "gpt2_small",
        "distilgpt2",
        "pythia_70m",
        "pythia_160m",
        "pythia_410m",
        "smollm_135m",
        "smollm2_135m",
        "smollm2_360m",
        "tinyllama_1_1b",
    ]
    out.extend(["## Modern Baseline Board", ""])
    out.extend(table(
        ["Model", "Params", "WikiText BPB", "LAMBADA BPB", "LAMBADA tok", "LAMBADA word", "MC avg norm", "Read"],
        [[
            models[mid]["name"],
            fmt_params(models[mid]["params"]),
            fmt_metric(get_run(by_suite, "modern_quality_v1", mid).get("wikitext_bpb"), "bpb"),
            fmt_metric(get_run(by_suite, "modern_quality_v1", mid).get("lambada_bpb"), "bpb"),
            fmt_percent(get_run(by_suite, "modern_quality_v1", mid).get("lambada_tok_acc")),
            fmt_percent(get_run(by_suite, "modern_quality_v1", mid).get("lambada_word_acc")),
            fmt_percent(get_run(by_suite, "modern_quality_v1", mid).get("mc_avg_norm")),
            models[mid]["notes"],
        ] for mid in modern_ids],
    ))
    out.append("")

    mc_ids = [
        "solollm_v4_2_150m",
        "solollm_v4_2_123m",
        "solollm_v3_150m",
        "solollm_v3_123m",
        "gpt2_small",
        "distilgpt2",
        "pythia_70m",
        "pythia_160m",
        "pythia_410m",
        "smollm_135m",
        "smollm2_135m",
        "smollm2_360m",
        "tinyllama_1_1b",
    ]
    out.extend(["## Multiple-Choice Detail", ""])
    out.extend(table(
        ["Model", "HellaSwag", "PIQA", "ARC-Easy", "ARC-Challenge", "WinoGrande", "Average"],
        [[
            models[mid]["name"],
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("hellaswag_norm")),
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("piqa_norm")),
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("arc_easy_norm")),
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("arc_challenge_norm")),
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("winogrande_norm")),
            fmt_percent(get_run(by_suite, "multiple_choice_v1", mid).get("mc_avg_norm")),
        ] for mid in mc_ids],
    ))
    out.append("")

    gen_ids = [
        "gpt2_small",
        "solollm_v3_123m",
        "solollm_v3_150m",
        "solollm_v4_2_123m",
        "solollm_v4_2_150m",
    ]
    out.extend(["## Generation Diagnostics", ""])
    out.extend(table(
        ["Model", "Distinct-1", "Distinct-2", "Repeated bigrams", "Repeated trigrams", "Bad loops"],
        [[
            models[mid]["name"],
            fmt_number(get_run(by_suite, "generation_v1", mid).get("distinct_1"), 4),
            fmt_number(get_run(by_suite, "generation_v1", mid).get("distinct_2"), 4),
            fmt_number(get_run(by_suite, "generation_v1", mid).get("repeated_bigrams"), 4),
            fmt_number(get_run(by_suite, "generation_v1", mid).get("repeated_trigrams"), 4),
            fmt_metric(get_run(by_suite, "generation_v1", mid).get("bad_loops"), "integer"),
        ] for mid in gen_ids],
    ))
    out.append("")

    infer_ids = [
        "solollm_v4_2_150m",
        "solollm_v4_2_123m",
        "gpt2_small",
        "pythia_160m",
        "pythia_410m",
        "smollm2_135m",
        "smollm2_360m",
        "tinyllama_1_1b",
    ]
    out.extend(["## Local Inference", ""])
    out.append("Batch size `1`, greedy generation, FP16, three fixed prompts.")
    out.append("")
    out.extend(table(
        ["Model", "Params", "Generation path", "Tok/s", "Peak VRAM"],
        [[
            models[mid]["name"],
            fmt_params(models[mid]["params"]),
            str(get_run(by_suite, "local_inference_v1", mid).get("generation_path", "-")),
            fmt_number(get_run(by_suite, "local_inference_v1", mid).get("tok_s")),
            fmt_metric(get_run(by_suite, "local_inference_v1", mid).get("peak_vram_gb"), "gb"),
        ] for mid in infer_ids],
    ))
    out.append("")

    proxy_ids = [
        "solollm_v4_2_proxy_57m_300m",
        "solollm_v5_proxy_57m_1b",
    ]
    out.extend(["## V5 Proxy Gate", ""])
    out.extend(table(
        ["Proxy", "Params", "Tokens", "Val PPL", "Train loss", "WikiText BPB", "LAMBADA BPB", "LAMBADA tok", "LAMBADA word", "MC avg norm"],
        [[
            models[mid]["name"],
            fmt_params(models[mid]["params"]),
            models[mid]["train_tokens_display"],
            fmt_number(get_run(by_suite, "proxy_gate_v1", mid).get("val_ppl")),
            fmt_number(get_run(by_suite, "proxy_gate_v1", mid).get("train_loss"), 4),
            fmt_metric(get_run(by_suite, "proxy_gate_v1", mid).get("wikitext_bpb"), "bpb"),
            fmt_metric(get_run(by_suite, "proxy_gate_v1", mid).get("lambada_bpb"), "bpb"),
            fmt_percent(get_run(by_suite, "proxy_gate_v1", mid).get("lambada_tok_acc")),
            fmt_percent(get_run(by_suite, "proxy_gate_v1", mid).get("lambada_word_acc")),
            fmt_percent(get_run(by_suite, "proxy_gate_v1", mid).get("mc_avg_norm")),
        ] for mid in proxy_ids],
    ))
    out.append("")

    out.extend([
        "## Claim Standard",
        "",
        "A leaderboard win means the model is better across the fixed suite shown here.",
        "It does not prove the model is better at every possible language task.",
        "",
        "The strongest current SoloLLM claim is that v4.2 150M beats GPT-2 small",
        "across the fixed project suite, while the smaller v4.2 123M checkpoint",
        "wins the normalized multiple-choice suite but not every metric.",
        "",
    ])

    return "\n".join(out)


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
