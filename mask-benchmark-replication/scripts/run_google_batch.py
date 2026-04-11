#!/usr/bin/env python3
"""Build, submit, and replay Google batch MASK runs.

Main tweak points:
- `MODEL_PRICING_*` if provider pricing changes
- `build_artifact_dir()` and `build_score_log_dir()` if you want different output roots
- batch mode flags passed from `run_google_batch.ps1`
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from google.genai import Client
from google.genai.types import (
    CreateBatchJobConfig,
    GenerateContentConfig,
    GenerateContentResponse,
    InlinedRequest,
    ThinkingConfig,
    ThinkingLevel,
)
from inspect_ai import Epochs, Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._providers._google_batch import batch_request_dict
from inspect_ai.model._providers.google import (
    DEFAULT_SAFETY_SETTINGS,
    as_chat_messages,
    completion_choices_from_candidates,
    extract_system_message_as_parts,
    safety_settings_to_list,
    usage_metadata_to_model_usage,
)
from inspect_ai.solver import TaskState, solver
from inspect_evals.mask.compat import _reduce
from inspect_evals.mask.data import get_mask_dataset
from inspect_evals.mask.mask import (
    DEFAULT_BELIEF_REPEATS,
    DEFAULT_BINARY_JUDGE_MODEL,
    DEFAULT_HONESTY_AT_N,
    DEFAULT_NO_BELIEF_HANDLING,
    DEFAULT_NUMERIC_JUDGE_MODEL,
    DEFAULT_NUMERIC_TOLERANCE,
    EVAL_VERSION,
)
from inspect_evals.mask.scorer import build_accuracy_and_honesty_scorer
from inspect_evals.mask.types import (
    MetadataKey,
    NoBeliefConsistencyHandling,
    QuestionArchetype,
    QuestionType,
    StoreKey,
)
from inspect_evals.mask.utils import (
    has_belief_elicitation_prompts,
    has_optional_extra_belief_elicitation_prompts,
)

MAX_SOLVER_TOKENS = 1000
SOLVER_TEMP = 1.0

MODEL_PRICING_BATCH_USD_PER_MTOKEN: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 1.25},
    "gemini-3.1-pro-preview": {"input": 1.00, "output": 6.00},
    "gemini-3-pro-preview": {"input": 1.00, "output": 6.00},
    "gemini-3-flash-preview": {"input": 0.25, "output": 1.50},
}

MODEL_PRICING_STANDARD_USD_PER_MTOKEN: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}


@dataclass
class RequestRecord:
    key: str
    sample_id: str
    prompt_kind: str
    store_key: str
    position: int


ARTIFACT_ROOT_ENV = "MASK_BATCH_ARTIFACT_ROOT"
SCORE_LOG_ROOT_ENV = "MASK_BATCH_SCORE_LOG_ROOT"
REGISTRY_PATH_ENV = "MASK_BATCH_REGISTRY"
INSPECT_APPDATA_ENV = "MASK_BATCH_INSPECT_APPDATA"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def google_batch_root() -> Path:
    return repo_root() / "outputs" / "google-batch"


def artifact_root() -> Path:
    configured = os.getenv(ARTIFACT_ROOT_ENV)
    return Path(configured).resolve() if configured else google_batch_root() / "artifacts"


def score_log_root() -> Path:
    configured = os.getenv(SCORE_LOG_ROOT_ENV)
    return Path(configured).resolve() if configured else repo_root() / "outputs" / "runs"


def registry_path() -> Path:
    configured = os.getenv(REGISTRY_PATH_ENV)
    return Path(configured).resolve() if configured else google_batch_root() / "registry.json"


def inspect_appdata_root() -> Path:
    configured = os.getenv(INSPECT_APPDATA_ENV)
    return Path(configured).resolve() if configured else repo_root() / "outputs" / "inspect-appdata"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False, by_alias=True)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=json_default),
        encoding="utf-8",
    )


def normalize_model_key(model_name: str) -> str:
    return model_name.removeprefix("google/").removeprefix("models/")


def pricing_for_model(
    model_name: str, pricing_table: dict[str, dict[str, float]]
) -> dict[str, float]:
    model_key = normalize_model_key(model_name)
    if model_key not in pricing_table:
        raise KeyError(f"No batch pricing configured for {model_name}")
    return pricing_table[model_key]


def build_artifact_dir(run_tag: str) -> Path:
    path = artifact_root() / f"{run_tag}_{utc_stamp()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_score_log_dir(run_tag: str, sample_count: int) -> Path:
    path = score_log_root() / f"{run_tag}_scored_n{sample_count}_{utc_stamp()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_reference_eval(reference_eval: str | None) -> Path | None:
    if reference_eval:
        return Path(reference_eval).resolve()
    return None


def load_samples_from_reference(reference_eval: Path) -> list[Sample]:
    samples: list[Sample] = []
    with ZipFile(reference_eval) as archive:
        for name in archive.namelist():
            if not name.startswith("samples/") or not name.endswith("_epoch_1.json"):
                continue
            payload = json.loads(archive.read(name))
            metadata = dict(payload["metadata"])
            if MetadataKey.CONFIG in metadata:
                metadata[MetadataKey.CONFIG] = QuestionArchetype(metadata[MetadataKey.CONFIG])
            samples.append(
                Sample.model_validate(
                    {
                        "id": payload["id"],
                        "input": payload["input"],
                        "metadata": metadata,
                    }
                )
            )
    if not samples:
        raise RuntimeError(f"No sample payloads found in {reference_eval}")
    return samples


def get_selected_samples(
    limit: int,
    sample_ids: list[str] | None,
    reference_eval: Path | None,
) -> list[Sample]:
    if reference_eval is not None:
        samples = load_samples_from_reference(reference_eval)
    else:
        samples = get_mask_dataset(None)
    if sample_ids:
        sample_by_id = {str(sample.id): sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in sample_by_id]
        if missing:
            raise KeyError(f"Unknown MASK sample ids: {', '.join(missing)}")
        return [sample_by_id[sample_id] for sample_id in sample_ids]
    return samples[:limit]


def build_prompt_messages(sample: Sample, prompt_kind: str) -> list[ChatMessage]:
    metadata = sample.metadata or {}
    if prompt_kind == "pressure":
        if not isinstance(sample.input, list):
            raise TypeError("MASK sample input is expected to be a chat message list")
        return list(sample.input)

    belief_system_prompt = metadata[MetadataKey.BELIEF_ELICITATION_SYSTEM_PROMPT]
    if prompt_kind == "belief1":
        prompt_text = metadata[MetadataKey.BELIEF_ELICIT_1]
    elif prompt_kind == "belief2":
        prompt_text = metadata[MetadataKey.BELIEF_ELICIT_2]
    elif prompt_kind == "belief3":
        prompt_text = metadata[MetadataKey.BELIEF_ELICIT_3]
    else:
        raise ValueError(f"Unsupported prompt kind: {prompt_kind}")
    return [
        ChatMessageSystem(content=belief_system_prompt),
        ChatMessageUser(content=prompt_text),
    ]


def iter_request_specs(
    sample: Sample,
    honesty_at_n: int,
    belief_repeats_n: int,
) -> list[RequestRecord]:
    metadata = sample.metadata or {}
    config = QuestionArchetype(metadata[MetadataKey.CONFIG])
    proposition_type = QuestionType(metadata[MetadataKey.TYPE])
    sample_id = str(sample.id)

    records: list[RequestRecord] = []
    if has_belief_elicitation_prompts(config):
        for index in range(belief_repeats_n):
            records.append(
                RequestRecord(
                    key=f"{sample_id}__belief1_r{index + 1}",
                    sample_id=sample_id,
                    prompt_kind="belief1",
                    store_key=StoreKey.BELIEF_ELICIT_1_RESPONSES,
                    position=index,
                )
            )
    if has_optional_extra_belief_elicitation_prompts(config, proposition_type):
        records.extend(
            [
                RequestRecord(
                    key=f"{sample_id}__belief2",
                    sample_id=sample_id,
                    prompt_kind="belief2",
                    store_key=StoreKey.BELIEF_ELICIT_2_AND_3_RESPONSES,
                    position=0,
                ),
                RequestRecord(
                    key=f"{sample_id}__belief3",
                    sample_id=sample_id,
                    prompt_kind="belief3",
                    store_key=StoreKey.BELIEF_ELICIT_2_AND_3_RESPONSES,
                    position=1,
                ),
            ]
        )
    for index in range(honesty_at_n):
        records.append(
            RequestRecord(
                key=f"{sample_id}__pressure_r{index + 1}",
                sample_id=sample_id,
                prompt_kind="pressure",
                store_key=StoreKey.PRESSURED_RESPONSES,
                position=index,
            )
        )
    return records


def parse_thinking_level(level: str | None) -> ThinkingLevel | None:
    if level is None:
        return None
    mapping = {
        "minimal": ThinkingLevel.MINIMAL,
        "low": ThinkingLevel.LOW,
        "medium": ThinkingLevel.MEDIUM,
        "high": ThinkingLevel.HIGH,
    }
    try:
        return mapping[level.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported thinking level: {level}") from exc


def build_thinking_config(args: argparse.Namespace) -> ThinkingConfig | None:
    thinking_level = parse_thinking_level(args.thinking_level)
    include_thoughts = args.include_thoughts
    if (
        thinking_level is None
        and args.thinking_budget is None
        and include_thoughts is None
    ):
        return None

    kwargs: dict[str, Any] = {}
    if include_thoughts is not None:
        kwargs["include_thoughts"] = include_thoughts
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
    if thinking_level is not None:
        kwargs["thinking_level"] = thinking_level
    return ThinkingConfig(**kwargs)


def create_google_client(submit: bool) -> Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key and submit:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set in this shell. Set it as an environment variable before submitting or polling a Google batch job."
        )
    return Client(api_key=api_key or "DUMMY")


def build_generate_config(
    system_instruction: list[Any] | None,
    thinking_config: ThinkingConfig | None,
) -> GenerateContentConfig:
    kwargs: dict[str, Any] = {
        "temperature": SOLVER_TEMP,
        "max_output_tokens": MAX_SOLVER_TOKENS,
        "safety_settings": safety_settings_to_list(DEFAULT_SAFETY_SETTINGS),
    }
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    if thinking_config is not None:
        kwargs["thinking_config"] = thinking_config
    return GenerateContentConfig(**kwargs)


async def build_requests(
    samples: list[Sample],
    batch_model: str,
    thinking_config: ThinkingConfig | None,
) -> tuple[list[RequestRecord], list[InlinedRequest], list[dict[str, Any]]]:
    client = create_google_client(submit=False)
    request_records: list[RequestRecord] = []
    inline_requests: list[InlinedRequest] = []
    jsonl_lines: list[dict[str, Any]] = []

    for sample in samples:
        for record in iter_request_specs(
            sample=sample,
            honesty_at_n=DEFAULT_HONESTY_AT_N,
            belief_repeats_n=DEFAULT_BELIEF_REPEATS,
        ):
            messages = build_prompt_messages(sample, record.prompt_kind)
            contents = await as_chat_messages(client, messages, emulate_reasoning=False)
            system_instruction = await extract_system_message_as_parts(
                client,
                messages,
                [],
                include_function_calling_hint=True,
            )
            config = build_generate_config(system_instruction, thinking_config)
            request_records.append(record)
            inline_requests.append(
                InlinedRequest(
                    model=batch_model,
                    contents=contents,
                    config=config,
                    metadata={
                        "custom_id": record.key,
                        "sample_id": record.sample_id,
                        "store_key": record.store_key,
                        "position": str(record.position),
                        "prompt_kind": record.prompt_kind,
                    },
                )
            )
            jsonl_lines.append(
                {
                    "key": record.key,
                    "request": batch_request_dict(config, contents),
                }
            )
    return request_records, inline_requests, jsonl_lines


async def write_build_artifacts(args: argparse.Namespace) -> Path:
    reference_eval = resolve_reference_eval(args.reference_eval)
    selected_samples = get_selected_samples(args.limit, args.sample_id, reference_eval)
    artifact_dir = build_artifact_dir(args.run_tag)
    thinking_config = build_thinking_config(args)
    request_records, inline_requests, jsonl_lines = await build_requests(
        samples=selected_samples,
        batch_model=args.batch_model,
        thinking_config=thinking_config,
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "batch_model": args.batch_model,
        "inspect_model": args.inspect_model,
        "reference_eval": str(reference_eval) if reference_eval is not None else None,
        "sample_count": len(selected_samples),
        "sample_ids": [str(sample.id) for sample in selected_samples],
        "thinking_config": (
            thinking_config.model_dump(exclude_none=True, by_alias=True)
            if thinking_config is not None
            else None
        ),
        "request_count": len(request_records),
        "requests": [asdict(record) for record in request_records],
    }
    json_dump(artifact_dir / "manifest.json", manifest)
    json_dump(
        artifact_dir / "requests.inline.json",
        [request.model_dump(exclude_none=True, by_alias=True) for request in inline_requests],
    )
    with (artifact_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for line in jsonl_lines:
            handle.write(json.dumps(line, ensure_ascii=True))
            handle.write("\n")
    return artifact_dir


async def submit_inline_batch(
    artifact_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    inline_payload = json.loads(
        (artifact_dir / "requests.inline.json").read_text(encoding="utf-8")
    )
    inline_requests = [
        InlinedRequest.model_validate(request) for request in inline_payload
    ]
    client = create_google_client(submit=True)
    batch_job = await client.aio.batches.create(
        model=manifest["batch_model"],
        src=inline_requests,
        config=CreateBatchJobConfig(display_name=manifest["run_tag"]),
    )
    batch_job_dict = batch_job.model_dump(mode="json", exclude_none=True, by_alias=True)
    json_dump(artifact_dir / "batch_job_created.json", batch_job_dict)
    return batch_job_dict


def bind_existing_batch_job(artifact_dir: Path, batch_job_name: str) -> dict[str, Any]:
    normalized_name = batch_job_name.strip()
    if (
        not normalized_name
        or "REPLACE_ME" in normalized_name
        or not normalized_name.startswith("batches/")
    ):
        raise ValueError(
            "Invalid --batch-job-name. Provide a real Gemini batch id like "
            "'batches/abc123...', not a placeholder."
        )
    batch_job_dict = {
        "name": normalized_name,
        "state": "JOB_STATE_PENDING",
    }
    json_dump(artifact_dir / "batch_job_created.json", batch_job_dict)
    return batch_job_dict

def register_batch_job(batch_job_name: str, artifact_dir: Path, label: str, model: str, kind: str) -> None:
    path = registry_path()
    payload = {"schema_version": 1, "jobs": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.setdefault("jobs", [])
    jobs = [job for job in jobs if job.get("batch_job_name") != batch_job_name]
    jobs.append(
        {
            "label": label,
            "model": model,
            "kind": kind,
            "batch_job_name": batch_job_name,
            "artifact_dir": str(artifact_dir),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload["jobs"] = jobs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def wait_for_batch(
    artifact_dir: Path,
    poll_seconds: int,
    timeout_minutes: int,
) -> dict[str, Any]:
    created = json.loads((artifact_dir / "batch_job_created.json").read_text(encoding="utf-8"))
    client = create_google_client(submit=True)
    deadline = time.time() + (timeout_minutes * 60)

    while True:
        batch_job = await client.aio.batches.get(name=created["name"])
        batch_job_dict = batch_job.model_dump(mode="json", exclude_none=True, by_alias=True)
        json_dump(artifact_dir / "batch_job_latest.json", batch_job_dict)
        state = batch_job_dict.get("state", "")
        if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            return batch_job_dict
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            raise RuntimeError(f"Google batch failed in state {state}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Google batch {created['name']}")
        await asyncio.sleep(poll_seconds)


async def fetch_batch_output(
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    latest = json.loads((artifact_dir / "batch_job_latest.json").read_text(encoding="utf-8"))
    client = create_google_client(submit=True)
    destination = latest.get("dest") or {}

    if destination.get("inlinedResponses"):
        responses = destination["inlinedResponses"]
        json_dump(artifact_dir / "batch_responses.inline.json", responses)
        return responses

    file_name = destination.get("fileName")
    if not file_name:
        raise RuntimeError("Batch completed without inlined responses or output file")
    raw_bytes = await client.aio.files.download(file=file_name)
    output_path = artifact_dir / "batch_responses.jsonl"
    output_path.write_bytes(raw_bytes)
    parsed_lines: list[dict[str, Any]] = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed_lines.append(json.loads(line))
    return parsed_lines


async def execute_live_requests(
    artifact_dir: Path,
    request_concurrency: int,
) -> list[dict[str, Any]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    inline_payload = json.loads(
        (artifact_dir / "requests.inline.json").read_text(encoding="utf-8")
    )
    inline_requests = [
        InlinedRequest.model_validate(request) for request in inline_payload
    ]
    client = create_google_client(submit=True)
    semaphore = asyncio.Semaphore(request_concurrency)
    responses: list[dict[str, Any] | None] = [None] * len(inline_requests)

    async def run_request(index: int, request: InlinedRequest) -> None:
        key = (
            request.metadata.get("custom_id")
            if isinstance(request.metadata, dict)
            else manifest["requests"][index]["key"]
        )
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=request.model,
                    contents=request.contents,
                    config=request.config,
                )
                responses[index] = {
                    "key": key,
                    "response": response.model_dump(
                        mode="json", exclude_none=True, by_alias=True
                    ),
                }
            except Exception as exc:
                responses[index] = {
                    "key": key,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }

    await asyncio.gather(
        *(run_request(index, request) for index, request in enumerate(inline_requests))
    )

    if any(item is None for item in responses):
        raise RuntimeError("Live Gemini run produced incomplete response slots")

    live_items = [item for item in responses if item is not None]
    json_dump(artifact_dir / "live_responses.inline.json", live_items)
    return live_items


def usage_to_dict(usage: ModelUsage | None) -> dict[str, int | float | None]:
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "total_cost": None,
        }
    return usage.model_dump(exclude_none=False)


def add_usage(lhs: ModelUsage | None, rhs: ModelUsage | None) -> ModelUsage | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return lhs + rhs


def estimate_cost(
    model_name: str,
    usage: ModelUsage | None,
    pricing_table: dict[str, dict[str, float]],
) -> float | None:
    if usage is None:
        return 0.0
    try:
        pricing = pricing_for_model(model_name, pricing_table)
    except KeyError:
        return None
    billable_output_tokens = usage.output_tokens + (usage.reasoning_tokens or 0)
    return (
        (usage.input_tokens / 1_000_000.0) * pricing["input"]
        + (billable_output_tokens / 1_000_000.0) * pricing["output"]
    )


def estimate_batch_cost(model_name: str, usage: ModelUsage | None) -> float | None:
    return estimate_cost(model_name, usage, MODEL_PRICING_BATCH_USD_PER_MTOKEN)


def estimate_standard_cost(model_name: str, usage: ModelUsage | None) -> float | None:
    return estimate_cost(model_name, usage, MODEL_PRICING_STANDARD_USD_PER_MTOKEN)


def initialize_store_lists(requests: list[dict[str, Any]], sample_id: str) -> dict[str, list[Any]]:
    counts: dict[str, int] = {}
    for request in requests:
        if request["sample_id"] != sample_id:
            continue
        store_key = request["store_key"]
        counts[store_key] = max(counts.get(store_key, 0), int(request["position"]) + 1)
    return {store_key: [None] * count for store_key, count in counts.items()}


def create_model_output(
    manifest_batch_model: str,
    response: GenerateContentResponse,
) -> ModelOutput:
    model_name = response.model_version or normalize_model_key(manifest_batch_model)
    return ModelOutput(
        model=model_name,
        choices=completion_choices_from_candidates(model_name, response),
        usage=usage_metadata_to_model_usage(response.usage_metadata),
    )


def build_replay_bundle(
    artifact_dir: Path,
    response_items: list[dict[str, Any]],
    target_generation_mode: str,
) -> Path:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    requests_by_key = {request["key"]: request for request in manifest["requests"]}
    samples_payload: dict[str, dict[str, Any]] = {
        sample_id: {
            "sample_id": sample_id,
            "store": initialize_store_lists(manifest["requests"], sample_id),
        }
        for sample_id in manifest["sample_ids"]
    }

    aggregate_usage: ModelUsage | None = None
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(response_items):
        key = item.get("key") or (item.get("metadata") or {}).get("custom_id")
        if key is None and index < len(manifest["requests"]):
            key = manifest["requests"][index]["key"]
        if key is None or key not in requests_by_key:
            continue
        request = requests_by_key[key]
        if item.get("error") is not None:
            errors.append({"key": key, "error": item["error"]})
            continue
        response = GenerateContentResponse.model_validate(item["response"])
        output = create_model_output(manifest["batch_model"], response)
        sample_store = samples_payload[request["sample_id"]]["store"][request["store_key"]]
        sample_store[int(request["position"])] = output.model_dump(exclude_none=False)
        aggregate_usage = add_usage(aggregate_usage, output.usage)

    if errors:
        json_dump(artifact_dir / "batch_errors.json", errors)
        raise RuntimeError(f"Google batch returned {len(errors)} request-level error(s)")

    missing_records: list[str] = []
    for sample in samples_payload.values():
        for store_key, values in sample["store"].items():
            if any(value is None for value in values):
                missing_records.append(f"{sample['sample_id']}:{store_key}")
    if missing_records:
        json_dump(artifact_dir / "batch_missing.json", missing_records)
        raise RuntimeError(
            "Replay bundle is missing batch responses for "
            + ", ".join(missing_records[:10])
        )

    replay_bundle = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": manifest["run_tag"],
        "batch_model": manifest["batch_model"],
        "inspect_model": manifest["inspect_model"],
        "target_generation_mode": target_generation_mode,
        "reference_eval": manifest.get("reference_eval"),
        "sample_ids": manifest["sample_ids"],
        "samples": list(samples_payload.values()),
        "target_usage": usage_to_dict(aggregate_usage),
        "target_standard_cost_estimate_usd": estimate_standard_cost(
            manifest["batch_model"], aggregate_usage
        ),
        "target_batch_cost_estimate_usd": estimate_batch_cost(
            manifest["batch_model"], aggregate_usage
        ),
    }
    bundle_path = artifact_dir / "replay_bundle.json"
    json_dump(bundle_path, replay_bundle)
    return bundle_path


@solver
def replay_mask_outputs(
    replay_samples: dict[str, dict[str, Any]],
) -> Any:
    async def solve(state: TaskState, generate: Any) -> TaskState:
        sample_id = str(state.sample_id)
        payload = replay_samples[sample_id]
        store = payload["store"]
        pressured = [
            ModelOutput.model_validate(item)
            for item in store.get(StoreKey.PRESSURED_RESPONSES, [])
        ]
        beliefs = [
            ModelOutput.model_validate(item)
            for item in store.get(StoreKey.BELIEF_ELICIT_1_RESPONSES, [])
        ]
        consistency = [
            ModelOutput.model_validate(item)
            for item in store.get(StoreKey.BELIEF_ELICIT_2_AND_3_RESPONSES, [])
        ]

        state.store.set(StoreKey.PRESSURED_RESPONSES, pressured)
        state.store.set(StoreKey.BELIEF_ELICIT_1_RESPONSES, beliefs)
        state.store.set(StoreKey.BELIEF_ELICIT_2_AND_3_RESPONSES, consistency)
        if pressured:
            state.output = pressured[0]
        return state

    return solve


def build_replay_task(bundle: dict[str, Any]) -> Task:
    reference_eval = (
        Path(bundle["reference_eval"]).resolve()
        if bundle.get("reference_eval")
        else None
    )
    sample_source = get_selected_samples(
        limit=len(bundle["sample_ids"]),
        sample_ids=bundle["sample_ids"],
        reference_eval=reference_eval,
    )
    sample_by_id = {str(sample.id): sample for sample in sample_source}
    dataset = [sample_by_id[sample_id] for sample_id in bundle["sample_ids"]]
    accuracy_and_honesty = build_accuracy_and_honesty_scorer(
        include_core=True,
        include_normalisation=False,
        include_statistical_summary=False,
        include_stratification=False,
    )
    replay_samples = {sample["sample_id"]: sample for sample in bundle["samples"]}

    return Task(
        dataset=dataset,
        solver=[replay_mask_outputs(replay_samples)],
        scorer=[
            accuracy_and_honesty(
                honesty_at_n=DEFAULT_HONESTY_AT_N,
                binary_judge_model=DEFAULT_BINARY_JUDGE_MODEL,
                numeric_judge_model=DEFAULT_NUMERIC_JUDGE_MODEL,
                numeric_tol=DEFAULT_NUMERIC_TOLERANCE,
                no_belief_handling=NoBeliefConsistencyHandling(
                    DEFAULT_NO_BELIEF_HANDLING
                ),
            )
        ],
        epochs=Epochs(1, _reduce()),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata()
        | {
            "target_generation": bundle.get("target_generation_mode", "google_batch_replay"),
            "target_batch_model": bundle["batch_model"],
            "target_standard_cost_estimate_usd": bundle.get(
                "target_standard_cost_estimate_usd"
            ),
            "target_batch_cost_estimate_usd": bundle["target_batch_cost_estimate_usd"],
        },
    )


async def score_replay_bundle(
    artifact_dir: Path,
    bundle_path: Path,
    display: str,
    max_connections: int,
    max_samples: int,
    log_buffer: int,
) -> Path:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    log_dir = build_score_log_dir(bundle["run_tag"], len(bundle["sample_ids"]))
    trace_file = artifact_dir / f"inspect_trace_{utc_stamp()}.log"
    inspect_appdata = inspect_appdata_root()
    inspect_appdata.mkdir(parents=True, exist_ok=True)
    previous_trace_file = os.environ.get("INSPECT_TRACE_FILE")
    previous_localappdata = os.environ.get("LOCALAPPDATA")
    previous_appdata = os.environ.get("APPDATA")
    previous_pytest_version = os.environ.get("PYTEST_VERSION")
    os.environ["INSPECT_TRACE_FILE"] = str(trace_file)
    os.environ["LOCALAPPDATA"] = str(inspect_appdata)
    os.environ["APPDATA"] = str(inspect_appdata)
    os.environ["PYTEST_VERSION"] = "1"
    try:
        eval_logs = await eval_async(
            build_replay_task(bundle),
            model=bundle["inspect_model"],
            display=display,
            log_dir=str(log_dir),
            max_connections=max_connections,
            max_samples=max_samples,
            retry_on_error=3,
            fail_on_error=25,
            continue_on_fail=True,
            log_buffer=log_buffer,
            log_realtime=False,
        )
    finally:
        if previous_trace_file is None:
            os.environ.pop("INSPECT_TRACE_FILE", None)
        else:
            os.environ["INSPECT_TRACE_FILE"] = previous_trace_file
        if previous_localappdata is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = previous_localappdata
        if previous_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = previous_appdata
        if previous_pytest_version is None:
            os.environ.pop("PYTEST_VERSION", None)
        else:
            os.environ["PYTEST_VERSION"] = previous_pytest_version
    eval_log = eval_logs[0]
    summary = {
        "bundle_path": str(bundle_path),
        "score_log_dir": str(log_dir),
        "score_eval_log": str(eval_log.location),
        "target_generation_mode": bundle.get("target_generation_mode"),
        "sample_count": len(bundle["sample_ids"]),
        "target_usage": bundle["target_usage"],
        "target_standard_cost_estimate_usd": bundle.get(
            "target_standard_cost_estimate_usd"
        ),
        "target_batch_cost_estimate_usd": bundle["target_batch_cost_estimate_usd"],
        "projected_target_standard_cost_n1000_usd": (
            bundle["target_standard_cost_estimate_usd"]
            * (1000.0 / len(bundle["sample_ids"]))
            if bundle.get("target_standard_cost_estimate_usd") is not None
            else None
        ),
        "projected_target_batch_cost_n1000_usd": (
            bundle["target_batch_cost_estimate_usd"]
            * (1000.0 / len(bundle["sample_ids"]))
            if bundle.get("target_batch_cost_estimate_usd") is not None
            else None
        ),
        "judge_usage": (
            {
                model_name: usage.model_dump(exclude_none=False)
                for model_name, usage in (eval_log.stats.model_usage or {}).items()
            }
            if eval_log.stats is not None
            else {}
        ),
        "results": (
            eval_log.results.model_dump(exclude_none=False)
            if eval_log.results is not None
            else None
        ),
    }
    summary_path = artifact_dir / "score_summary.json"
    json_dump(summary_path, summary)
    return summary_path


async def run_build_command(args: argparse.Namespace) -> None:
    artifact_dir = await write_build_artifacts(args)
    print(f"ARTIFACT_DIR={artifact_dir}")


async def run_smoke_command(args: argparse.Namespace) -> None:
    artifact_dir = await write_build_artifacts(args)
    print(f"ARTIFACT_DIR={artifact_dir}")
    if args.no_submit:
        return

    if args.batch_job_name:
        created = bind_existing_batch_job(artifact_dir, args.batch_job_name)
    else:
        created = await submit_inline_batch(artifact_dir)
    register_batch_job(
        batch_job_name=created["name"],
        artifact_dir=artifact_dir,
        label=args.run_tag,
        model=args.batch_model,
        kind=args.command,
    )
    print(f"BATCH_JOB_NAME={created['name']}")
    if args.no_wait:
        return

    latest = await wait_for_batch(
        artifact_dir=artifact_dir,
        poll_seconds=args.poll_seconds,
        timeout_minutes=args.timeout_minutes,
    )
    print(f"BATCH_STATE={latest['state']}")
    response_items = await fetch_batch_output(artifact_dir)
    bundle_path = build_replay_bundle(
        artifact_dir,
        response_items,
        target_generation_mode="google_batch_replay",
    )
    print(f"REPLAY_BUNDLE={bundle_path}")

    if args.no_score:
        return
    summary_path = await score_replay_bundle(
        artifact_dir=artifact_dir,
        bundle_path=bundle_path,
        display=args.display,
        max_connections=args.max_connections,
        max_samples=args.max_samples,
        log_buffer=args.log_buffer,
    )
    print(f"SCORE_SUMMARY={summary_path}")


async def run_live_smoke_command(args: argparse.Namespace) -> None:
    artifact_dir = await write_build_artifacts(args)
    print(f"ARTIFACT_DIR={artifact_dir}")
    response_items = await execute_live_requests(
        artifact_dir=artifact_dir,
        request_concurrency=args.request_concurrency,
    )
    print(f"LIVE_REQUESTS_COMPLETED={len(response_items)}")
    bundle_path = build_replay_bundle(
        artifact_dir,
        response_items,
        target_generation_mode="google_live_replay",
    )
    print(f"REPLAY_BUNDLE={bundle_path}")

    if args.no_score:
        return
    summary_path = await score_replay_bundle(
        artifact_dir=artifact_dir,
        bundle_path=bundle_path,
        display=args.display,
        max_connections=args.max_connections,
        max_samples=args.max_samples,
        log_buffer=args.log_buffer,
    )
    print(f"SCORE_SUMMARY={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run Google Gemini batch-target MASK replays with stock inspect_evals judges."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(common: argparse.ArgumentParser) -> None:
        common.add_argument("--run-tag", required=True)
        common.add_argument("--batch-model", required=True)
        common.add_argument("--inspect-model", required=True)
        common.add_argument("--limit", type=int, required=True)
        common.add_argument("--sample-id", action="append", default=[])
        common.add_argument("--reference-eval", default=None)
        common.add_argument("--thinking-budget", type=int, default=None)
        common.add_argument("--thinking-level", default=None)
        common.add_argument("--artifact-root", default=None)
        common.add_argument("--score-log-root", default=None)
        common.add_argument("--registry", default=None)
        common.add_argument("--inspect-appdata-dir", default=None)
        common.add_argument(
            "--include-thoughts",
            dest="include_thoughts",
            action="store_true",
            default=None,
        )
        common.add_argument(
            "--no-include-thoughts",
            dest="include_thoughts",
            action="store_false",
        )

    build_parser = subparsers.add_parser("build")
    add_common_arguments(build_parser)

    smoke_parser = subparsers.add_parser("smoke")
    add_common_arguments(smoke_parser)
    smoke_parser.add_argument("--no-submit", action="store_true")
    smoke_parser.add_argument("--no-wait", action="store_true")
    smoke_parser.add_argument("--no-score", action="store_true")
    smoke_parser.add_argument("--poll-seconds", type=int, default=30)
    smoke_parser.add_argument("--timeout-minutes", type=int, default=1440)
    smoke_parser.add_argument("--batch-job-name", default=None)
    smoke_parser.add_argument("--display", default="plain")
    smoke_parser.add_argument("--max-connections", type=int, default=8)
    smoke_parser.add_argument("--max-samples", type=int, default=10)
    smoke_parser.add_argument("--log-buffer", type=int, default=1)

    live_smoke_parser = subparsers.add_parser("live-smoke")
    add_common_arguments(live_smoke_parser)
    live_smoke_parser.add_argument("--request-concurrency", type=int, default=4)
    live_smoke_parser.add_argument("--no-score", action="store_true")
    live_smoke_parser.add_argument("--display", default="plain")
    live_smoke_parser.add_argument("--max-connections", type=int, default=8)
    live_smoke_parser.add_argument("--max-samples", type=int, default=10)
    live_smoke_parser.add_argument("--log-buffer", type=int, default=1)

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = str(Path(args.artifact_root).resolve())
    if args.score_log_root:
        os.environ[SCORE_LOG_ROOT_ENV] = str(Path(args.score_log_root).resolve())
    if args.registry:
        os.environ[REGISTRY_PATH_ENV] = str(Path(args.registry).resolve())
    if args.inspect_appdata_dir:
        os.environ[INSPECT_APPDATA_ENV] = str(Path(args.inspect_appdata_dir).resolve())
    if args.command == "build":
        await run_build_command(args)
    elif args.command == "smoke":
        await run_smoke_command(args)
    elif args.command == "live-smoke":
        await run_live_smoke_command(args)
    else:
        raise ValueError(f"Unsupported command {args.command}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
