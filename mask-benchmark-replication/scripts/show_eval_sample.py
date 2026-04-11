#!/usr/bin/env python3
"""Print a compact summary and one example sample from an Inspect `.eval` bundle.

This helper also works with the sanitized example bundle shipped under
`examples/smoke-success/`.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/show_eval_sample.py <path-to-eval>", file=sys.stderr)
        return 2

    eval_path = Path(sys.argv[1])
    if not eval_path.exists():
        print(f"File not found: {eval_path}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(eval_path) as bundle:
        header = json.loads(bundle.read("header.json"))
        reductions = json.loads(bundle.read("reductions.json"))
        sample_names = sorted(name for name in bundle.namelist() if name.startswith("samples/"))
        sample_records = [(name, json.loads(bundle.read(name))) for name in sample_names]
        sample_name, sample = next(
            (
                (name, record)
                for name, record in sample_records
                if record.get("metadata", {}).get("config") == "known_facts"
                and record.get("metadata", {}).get("example_case") == "honest"
            ),
            sample_records[0],
        )
        sample_metas = [record.get("metadata", {}) for _, record in sample_records]

    reduced_samples = reductions[0].get("samples", []) if reductions else []
    accuracy_counts = Counter()
    honesty_counts = Counter()
    config_counts = Counter()
    case_counts = Counter()
    for item in reduced_samples:
        value = item.get("value", {})
        accuracy = value.get("accuracy")
        honesty = value.get("honesty")
        if accuracy is not None:
            accuracy_counts[accuracy] += 1
        if honesty is not None:
            honesty_counts[honesty] += 1
    for sample_meta in sample_metas:
        config = sample_meta.get("config")
        case = sample_meta.get("example_case")
        if config is not None:
            config_counts[config] += 1
        if case is not None:
            case_counts[case] += 1

    total_reduced = len(reduced_samples)
    summary = {
        "total_reduced_samples": total_reduced,
        "accuracy_counts": dict(accuracy_counts),
        "honesty_counts": dict(honesty_counts),
        "config_counts": dict(config_counts),
        "example_case_counts": dict(case_counts),
    }
    if total_reduced:
        summary["accuracy_correct_rate"] = round(accuracy_counts.get("correct", 0) / total_reduced, 3)
        summary["honesty_honest_rate"] = round(honesty_counts.get("honest", 0) / total_reduced, 3)

    print("status:", header["status"])
    print("dataset_samples:", header["eval"]["dataset"]["samples"])
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("sample_file:", sample_name)
    print("sample_id:", sample["id"])
    print("sample_config:", sample.get("metadata", {}).get("config"))
    print("sample_case:", sample.get("metadata", {}).get("example_case"))
    print("sample_input_1:", sample["input"][0]["content"][:700])
    print("sample_output:", json.dumps(sample.get("output"), ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
