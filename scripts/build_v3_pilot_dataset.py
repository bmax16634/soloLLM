#!/usr/bin/env python
"""Build the SoloLLM v3 1024-token pilot corpus from streaming datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_int(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    return int(str(value).replace("_", ""))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def split_for_hash(doc_hash: str, split_probs: dict[str, float]) -> str:
    value = int(doc_hash[:16], 16) / float(16**16)
    train_cut = split_probs["train"]
    val_cut = train_cut + split_probs["val"]
    if value < train_cut:
        return "train"
    if value < val_cut:
        return "val"
    return "test"


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for char in text if char.isalpha())
    return alpha / max(len(text), 1)


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts = Counter(lines)
    repeated_chars = sum(len(line) * count for line, count in counts.items() if count > 1)
    total_chars = sum(len(line) for line in lines)
    return repeated_chars / max(total_chars, 1)


def first_text_field(example: dict[str, Any], preferred: str | None) -> str | None:
    if preferred and isinstance(example.get(preferred), str):
        return example[preferred]
    for value in example.values():
        if isinstance(value, str) and len(value) > 0:
            return value
    return None


def passes_filters(
    text: str,
    example: dict[str, Any],
    filters: dict[str, Any],
) -> tuple[bool, str]:
    min_chars = int(filters.get("min_chars", 0))
    max_chars = int(filters.get("max_chars", 10**9))
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"

    language = example.get("language")
    if isinstance(language, str) and language and language.lower() not in {"en", "eng", "english"}:
        return False, "non_english"

    min_language_score = filters.get("min_language_score")
    if min_language_score is not None:
        score = example.get("language_score")
        if isinstance(score, (int, float)) and float(score) < float(min_language_score):
            return False, "low_language_score"

    min_alpha = float(filters.get("min_alpha_ratio", 0.0))
    if alpha_ratio(text) < min_alpha:
        return False, "low_alpha_ratio"

    max_repeat = float(filters.get("max_repeated_line_fraction", 1.0))
    if repeated_line_fraction(text) > max_repeat:
        return False, "repeated_lines"

    return True, "accepted"


@dataclass
class SplitWriter:
    name: str
    output_dir: Path
    seq_length: int
    shard_tokens: int
    dtype: torch.dtype
    buffer: list[int] = field(default_factory=list)
    shard_paths: list[Path] = field(default_factory=list)
    rows: int = 0
    tokens: int = 0

    def add(self, token_ids: list[int]) -> None:
        self.buffer.extend(token_ids)
        while len(self.buffer) >= self.shard_tokens:
            self.flush(final=False)

    def flush(self, *, final: bool) -> None:
        usable = (len(self.buffer) // self.seq_length) * self.seq_length
        if not usable:
            return
        if not final:
            usable = min(usable, self.shard_tokens)
            usable = (usable // self.seq_length) * self.seq_length
        block = self.buffer[:usable]
        self.buffer = self.buffer[usable:]
        tensor = torch.tensor(block, dtype=self.dtype).view(-1, self.seq_length)
        path = self.output_dir / self.name / f"shard_{len(self.shard_paths):05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(tensor, tmp_path)
        os.replace(tmp_path, path)
        self.shard_paths.append(path)
        self.rows += int(tensor.size(0))
        self.tokens += int(tensor.numel())


@dataclass
class SourceState:
    id: str
    config: dict[str, Any]
    quota_tokens: int
    iterator: Iterator[dict[str, Any]]
    accepted_tokens: int = 0
    docs_seen: int = 0
    docs_accepted: int = 0
    exhausted: bool = False
    filter_counts: Counter[str] = field(default_factory=Counter)

    @property
    def remaining(self) -> int:
        return max(self.quota_tokens - self.accepted_tokens, 0)


def make_dataset_iterator(source: dict[str, Any], *, shuffle_buffer: int, seed: int) -> Iterator[dict[str, Any]]:
    dataset = source["dataset"]
    config = source.get("config")
    split = source.get("split", "train")
    if config is None:
        stream = load_dataset(dataset, split=split, streaming=True)
    else:
        stream = load_dataset(dataset, config, split=split, streaming=True)
    if shuffle_buffer > 0:
        stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return iter(stream)


def normalize_source_weights(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_weight = sum(float(source.get("weight", 0.0)) for source in sources)
    if total_weight <= 0:
        raise ValueError("at least one enabled source must have positive weight")
    for source in sources:
        source["normalized_weight"] = float(source.get("weight", 0.0)) / total_weight
    return sources


def enabled_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [source for source in manifest.get("sources", []) if source.get("enabled", True)]
    return normalize_source_weights(sources)


def choose_source(rng: random.Random, states: list[SourceState]) -> SourceState | None:
    active = [state for state in states if not state.exhausted and state.remaining > 0]
    if not active:
        return None
    weights = [state.remaining for state in active]
    return rng.choices(active, weights=weights, k=1)[0]


def link_combined_chunks(output_dir: Path, writers: dict[str, SplitWriter]) -> dict[str, Any]:
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob("shard_*.pt"):
        old.unlink()

    ranges: dict[str, Any] = {}
    next_index = 0
    for split_name in ["train", "val", "test"]:
        paths = writers[split_name].shard_paths
        start = next_index
        for path in paths:
            dest = chunks_dir / f"shard_{next_index:05d}.pt"
            try:
                os.link(path, dest)
            except OSError:
                shutil.copy2(path, dest)
            next_index += 1
        end = next_index - 1
        ranges[split_name] = {
            "start": start,
            "end": end,
            "range": f"{start}:{end}" if paths else None,
            "shards": len(paths),
            "tokens": writers[split_name].tokens,
            "rows": writers[split_name].rows,
        }
    return {"chunks_dir": str(chunks_dir), "shard_ranges": ranges}


def build(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_yaml(manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_tokens = parse_int(args.target_tokens) or int(manifest["target_tokens"])
    seq_length = int(args.seq_length or manifest.get("seq_length", 1024))
    shard_tokens = parse_int(args.shard_tokens) or int(manifest.get("shard_tokens", 8_388_608))
    shard_tokens = max(seq_length, (shard_tokens // seq_length) * seq_length)
    seed = int(args.seed if args.seed is not None else manifest.get("streaming", {}).get("seed", 1337))
    shuffle_buffer = int(args.shuffle_buffer if args.shuffle_buffer is not None else manifest.get("streaming", {}).get("shuffle_buffer", 0))
    dtype = torch.int64 if args.dtype == "int64" else torch.int32

    split_probs = {key: float(value) for key, value in manifest.get("split", {}).items()}
    total_split = sum(split_probs.values())
    split_probs = {key: value / total_split for key, value in split_probs.items()}
    for key in ["train", "val", "test"]:
        split_probs.setdefault(key, 0.0)

    tokenizer_name = args.tokenizer or manifest.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    eos_id = int(tokenizer.eos_token_id)

    sources = enabled_sources(manifest)
    if args.sources:
        selected = set(args.sources)
        sources = [source for source in sources if str(source["id"]) in selected]
        missing = selected - {str(source["id"]) for source in sources}
        if missing:
            raise ValueError(f"unknown or disabled source ids: {sorted(missing)}")
        if not sources:
            raise ValueError("--sources selected no enabled sources")
        sources = normalize_source_weights(sources)
    rng = random.Random(seed)
    states: list[SourceState] = []
    for index, source in enumerate(sources):
        quota = int(math.ceil(target_tokens * float(source["normalized_weight"])))
        iterator = make_dataset_iterator(source, shuffle_buffer=shuffle_buffer, seed=seed + index)
        states.append(SourceState(id=str(source["id"]), config=source, quota_tokens=quota, iterator=iterator))

    writers = {
        split: SplitWriter(split, output_dir, seq_length, shard_tokens, dtype)
        for split in ["train", "val", "test"]
    }
    seen_hashes: set[str] = set()
    global_counts: Counter[str] = Counter()
    start_time = time.time()
    docs_seen = 0
    accepted_docs = 0
    accepted_tokens = 0

    run_meta = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "target_tokens": target_tokens,
        "seq_length": seq_length,
        "shard_tokens": shard_tokens,
        "tokenizer": tokenizer_name,
        "dtype": str(dtype),
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "started_at": start_time,
        "sources": [
            {
                "id": state.id,
                "dataset": state.config["dataset"],
                "config": state.config.get("config"),
                "weight": state.config.get("weight"),
                "normalized_weight": state.config.get("normalized_weight"),
                "quota_tokens": state.quota_tokens,
            }
            for state in states
        ],
    }
    write_json(output_dir / "run_meta.json", run_meta)

    while accepted_tokens < target_tokens:
        state = choose_source(rng, states)
        if state is None:
            break
        try:
            example = next(state.iterator)
        except StopIteration:
            state.exhausted = True
            continue

        docs_seen += 1
        state.docs_seen += 1
        text = first_text_field(example, state.config.get("text_field"))
        if text is None:
            state.filter_counts["missing_text"] += 1
            global_counts["missing_text"] += 1
            continue
        text = normalize_text(text)
        ok, reason = passes_filters(text, example, manifest.get("filters", {}))
        if not ok:
            state.filter_counts[reason] += 1
            global_counts[reason] += 1
            continue
        doc_hash = stable_hash(text)
        if doc_hash in seen_hashes:
            state.filter_counts["exact_duplicate"] += 1
            global_counts["exact_duplicate"] += 1
            continue
        seen_hashes.add(doc_hash)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            state.filter_counts["empty_tokens"] += 1
            global_counts["empty_tokens"] += 1
            continue
        token_ids.append(eos_id)

        split = split_for_hash(doc_hash, split_probs)
        writers[split].add(token_ids)
        token_count = len(token_ids)
        state.accepted_tokens += token_count
        state.docs_accepted += 1
        accepted_docs += 1
        accepted_tokens += token_count
        global_counts["accepted"] += 1

        if args.max_docs and docs_seen >= args.max_docs:
            break
        if accepted_docs % int(args.stats_every_docs) == 0:
            stats = build_stats(
                states,
                writers,
                global_counts,
                docs_seen=docs_seen,
                accepted_docs=accepted_docs,
                accepted_tokens=accepted_tokens,
                target_tokens=target_tokens,
                start_time=start_time,
            )
            write_json(output_dir / "build_stats.json", stats)
            print(json.dumps({"progress": stats["progress"], "splits": stats["splits"]}), flush=True)

    for writer in writers.values():
        writer.flush(final=True)

    combined = link_combined_chunks(output_dir, writers)
    stats = build_stats(
        states,
        writers,
        global_counts,
        docs_seen=docs_seen,
        accepted_docs=accepted_docs,
        accepted_tokens=accepted_tokens,
        target_tokens=target_tokens,
        start_time=start_time,
    )
    stats.update(combined)
    stats["finished_at"] = time.time()
    stats["elapsed_sec"] = stats["finished_at"] - start_time
    write_json(output_dir / "build_stats.json", stats)
    write_json(output_dir / "splits.json", combined)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


def build_stats(
    states: list[SourceState],
    writers: dict[str, SplitWriter],
    global_counts: Counter[str],
    *,
    docs_seen: int,
    accepted_docs: int,
    accepted_tokens: int,
    target_tokens: int,
    start_time: float,
) -> dict[str, Any]:
    elapsed = max(time.time() - start_time, 1e-9)
    return {
        "progress": {
            "docs_seen": docs_seen,
            "accepted_docs": accepted_docs,
            "accepted_tokens": accepted_tokens,
            "target_tokens": target_tokens,
            "accepted_token_ratio": accepted_tokens / max(target_tokens, 1),
            "tokens_per_sec": accepted_tokens / elapsed,
            "elapsed_sec": elapsed,
        },
        "filters": dict(global_counts),
        "sources": {
            state.id: {
                "quota_tokens": state.quota_tokens,
                "accepted_tokens": state.accepted_tokens,
                "docs_seen": state.docs_seen,
                "docs_accepted": state.docs_accepted,
                "remaining_tokens": state.remaining,
                "exhausted": state.exhausted,
                "filters": dict(state.filter_counts),
            }
            for state in states
        },
        "splits": {
            name: {
                "tokens": writer.tokens + (len(writer.buffer) // writer.seq_length) * writer.seq_length,
                "flushed_tokens": writer.tokens,
                "buffer_tokens": len(writer.buffer),
                "rows": writer.rows,
                "shards": len(writer.shard_paths),
            }
            for name, writer in writers.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a streaming v3 pilot tokenized corpus")
    parser.add_argument("--manifest", default="sologpt_v3/data_sources.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-tokens", default=None, help="Total accepted tokens to build")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--shard-tokens", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=None)
    parser.add_argument("--sources", nargs="+", default=None, help="Restrict build to these manifest source ids")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--stats-every-docs", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return build(args)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
