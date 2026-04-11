#!/usr/bin/env python3
"""Estimate MASK target and judge costs from token counts or observed runtime profiles.

The main tweak points are the input price flags and, when helpful, the `run_profile_label`
loaded from `data/runtime_profiles.csv`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_profile(run_profile_label: str) -> dict[str, str]:
    path = project_root() / "data" / "runtime_profiles.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_profile_label") == run_profile_label:
                return row
            if row.get("profile_id") == run_profile_label:
                return row
    raise KeyError(f"Unknown run_profile_label: {run_profile_label}")


def price(tokens: float, usd_per_million: float) -> float:
    return (tokens / 1_000_000.0) * usd_per_million


def scaled(value: float, scale_from: int | None, scale_to: int | None) -> float:
    if not scale_from or not scale_to:
        return value
    return value * (float(scale_to) / float(scale_from))


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate MASK target plus judge costs.")
    parser.add_argument("--run-profile-label", default=None)
    parser.add_argument("--profile", default=None, help="Backward-compatible alias for --run-profile-label")
    parser.add_argument("--scale-from-samples", type=int, default=None)
    parser.add_argument("--scale-to-samples", type=int, default=None)
    parser.add_argument("--target-input-tokens", type=float, default=0.0)
    parser.add_argument("--target-output-tokens", type=float, default=0.0)
    parser.add_argument("--binary-judge-input-tokens", type=float, default=0.0)
    parser.add_argument("--binary-judge-output-tokens", type=float, default=0.0)
    parser.add_argument("--numeric-judge-input-tokens", type=float, default=0.0)
    parser.add_argument("--numeric-judge-output-tokens", type=float, default=0.0)
    parser.add_argument("--target-input-price", type=float, required=True)
    parser.add_argument("--target-output-price", type=float, required=True)
    parser.add_argument("--binary-judge-input-price", type=float, required=True)
    parser.add_argument("--binary-judge-output-price", type=float, required=True)
    parser.add_argument("--numeric-judge-input-price", type=float, required=True)
    parser.add_argument("--numeric-judge-output-price", type=float, required=True)
    args = parser.parse_args()

    selected_profile = args.run_profile_label or args.profile
    if selected_profile:
        profile = load_profile(selected_profile)
        args.scale_from_samples = args.scale_from_samples or int(profile["samples"])
        args.target_input_tokens = float(profile.get("target_tokens") or 0)
        args.binary_judge_input_tokens = float(profile.get("binary_judge_tokens") or profile.get("gpt4o_judge_tokens") or 0)
        args.numeric_judge_input_tokens = float(profile.get("numeric_judge_tokens") or profile.get("o3mini_judge_tokens") or 0)

    target_input = scaled(args.target_input_tokens, args.scale_from_samples, args.scale_to_samples)
    target_output = scaled(args.target_output_tokens, args.scale_from_samples, args.scale_to_samples)
    binary_input = scaled(args.binary_judge_input_tokens, args.scale_from_samples, args.scale_to_samples)
    binary_output = scaled(args.binary_judge_output_tokens, args.scale_from_samples, args.scale_to_samples)
    numeric_input = scaled(args.numeric_judge_input_tokens, args.scale_from_samples, args.scale_to_samples)
    numeric_output = scaled(args.numeric_judge_output_tokens, args.scale_from_samples, args.scale_to_samples)

    target_cost = price(target_input, args.target_input_price) + price(target_output, args.target_output_price)
    binary_cost = price(binary_input, args.binary_judge_input_price) + price(binary_output, args.binary_judge_output_price)
    numeric_cost = price(numeric_input, args.numeric_judge_input_price) + price(numeric_output, args.numeric_judge_output_price)
    total_cost = target_cost + binary_cost + numeric_cost

    print(f"target_cost_usd={target_cost:.4f}")
    print(f"binary_judge_cost_usd={binary_cost:.4f}")
    print(f"numeric_judge_cost_usd={numeric_cost:.4f}")
    print(f"total_cost_usd={total_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
