#!/usr/bin/env python3
"""Estimate MASK runtime from provider rate limits.

The main tweak points are the request-per-sample and token-per-sample values taken from
one of your own smoke runs.
"""

from __future__ import annotations

import argparse


def throughput(requests_per_sample: float, rpm: float, tokens_per_sample: float, tpm: float) -> tuple[float, str]:
    request_capacity = rpm / requests_per_sample if requests_per_sample > 0 else float("inf")
    token_capacity = tpm / tokens_per_sample if tokens_per_sample > 0 else float("inf")
    limiting = min(request_capacity, token_capacity)
    limiter = "requests" if request_capacity <= token_capacity else "tokens"
    return limiting, limiter


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate MASK runtime from provider rate limits.")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--target-requests-per-sample", type=float, required=True)
    parser.add_argument("--target-tokens-per-sample", type=float, required=True)
    parser.add_argument("--target-rpm", type=float, required=True)
    parser.add_argument("--target-tpm", type=float, required=True)
    parser.add_argument("--binary-judge-requests-per-sample", type=float, required=True)
    parser.add_argument("--binary-judge-tokens-per-sample", type=float, required=True)
    parser.add_argument("--binary-judge-rpm", type=float, required=True)
    parser.add_argument("--binary-judge-tpm", type=float, required=True)
    parser.add_argument("--numeric-judge-requests-per-sample", type=float, required=True)
    parser.add_argument("--numeric-judge-tokens-per-sample", type=float, required=True)
    parser.add_argument("--numeric-judge-rpm", type=float, required=True)
    parser.add_argument("--numeric-judge-tpm", type=float, required=True)
    args = parser.parse_args()

    target_spsm, target_limit = throughput(
        args.target_requests_per_sample,
        args.target_rpm,
        args.target_tokens_per_sample,
        args.target_tpm,
    )
    binary_spsm, binary_limit = throughput(
        args.binary_judge_requests_per_sample,
        args.binary_judge_rpm,
        args.binary_judge_tokens_per_sample,
        args.binary_judge_tpm,
    )
    numeric_spsm, numeric_limit = throughput(
        args.numeric_judge_requests_per_sample,
        args.numeric_judge_rpm,
        args.numeric_judge_tokens_per_sample,
        args.numeric_judge_tpm,
    )

    bottleneck_spsm = min(target_spsm, binary_spsm, numeric_spsm)
    estimated_minutes = args.samples / bottleneck_spsm if bottleneck_spsm > 0 else float("inf")
    recommended_max_connections = max(1, int(bottleneck_spsm * 0.8))
    recommended_max_samples = max(10, recommended_max_connections * 2)

    print(f"target_samples_per_minute={target_spsm:.2f}")
    print(f"target_limiter={target_limit}")
    print(f"binary_judge_samples_per_minute={binary_spsm:.2f}")
    print(f"binary_judge_limiter={binary_limit}")
    print(f"numeric_judge_samples_per_minute={numeric_spsm:.2f}")
    print(f"numeric_judge_limiter={numeric_limit}")
    print(f"estimated_min_runtime_minutes={estimated_minutes:.2f}")
    print(f"recommended_max_connections={recommended_max_connections}")
    print(f"recommended_max_samples={recommended_max_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
