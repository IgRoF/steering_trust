#!/usr/bin/env python3
"""Summarize an Inspect `.eval` log as JSON.

This works on genuine run logs and on the sanitized example bundle shipped
under `examples/smoke-success/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _model_usage(stats: Any) -> dict[str, dict[str, Any]]:
    usage = getattr(stats, "model_usage", None) if stats is not None else None
    if not usage:
        return {}

    payload: dict[str, dict[str, Any]] = {}
    for model, item in usage.items():
        payload[str(model)] = {
            "input_tokens": getattr(item, "input_tokens", None),
            "output_tokens": getattr(item, "output_tokens", None),
            "total_tokens": getattr(item, "total_tokens", None),
            "reasoning_tokens": getattr(item, "reasoning_tokens", None),
            "input_tokens_cache_write": getattr(item, "input_tokens_cache_write", None),
            "input_tokens_cache_read": getattr(item, "input_tokens_cache_read", None),
            "total_cost": getattr(item, "total_cost", None),
        }
    return payload


def _scores(results: Any) -> dict[str, Any]:
    if results is None:
        return {}

    scores = getattr(results, "scores", None)
    if not scores:
        return {}

    payload: dict[str, Any] = {}
    for score in scores:
        name = getattr(score, "name", None)
        if not name:
            continue
        value = getattr(score, "value", None)
        payload[str(name)] = _scalar(value)
    return payload


def summarize(path: Path) -> dict[str, Any]:
    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return {
            "path": str(path),
            "status": "read_error",
            "sample_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": None,
            "completed_at": None,
            "model": None,
            "task": None,
            "task_args": {},
            "scores": {},
            "model_usage": {},
        }
    samples = getattr(log, "samples", None) or []
    stats = getattr(log, "stats", None)
    eval_spec = getattr(log, "eval", None)
    task_args = getattr(eval_spec, "task_args", {}) or {}

    return {
        "path": str(path),
        "status": _scalar(getattr(log, "status", None)),
        "sample_count": len(samples),
        "error": repr(getattr(log, "error", None)) if getattr(log, "error", None) else None,
        "started_at": _scalar(getattr(stats, "started_at", None)),
        "completed_at": _scalar(getattr(stats, "completed_at", None)),
        "model": _scalar(getattr(eval_spec, "model", None)),
        "task": _scalar(getattr(eval_spec, "task", None)),
        "task_args": task_args,
        "scores": _scores(getattr(log, "results", None)),
        "model_usage": _model_usage(stats),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_eval_log.py <path-to.eval>", file=sys.stderr)
        return 2
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
