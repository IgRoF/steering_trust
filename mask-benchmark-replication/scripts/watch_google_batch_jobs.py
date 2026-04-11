#!/usr/bin/env python3
"""Poll Google batch jobs and optionally send Telegram notifications.

Main tweak points:
- `DEFAULT_REGISTRY` and `DEFAULT_STATE` for different local storage paths
- `DEFAULT_BOT_ENV` and `DEFAULT_CHAT_ENV` if you use different env var names
- `poll_once()` if you want different terminal-state messaging
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google.genai import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "outputs" / "google-batch" / "registry.json"
DEFAULT_STATE = PROJECT_ROOT / "outputs" / "google-batch" / "watch_state.json"
DEFAULT_BOT_ENV = "TELEGRAM_BOT_TOKEN"
DEFAULT_CHAT_ENV = "TELEGRAM_CHAT_ID"

SUCCESS_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
FAILURE_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
TERMINAL_STATES = SUCCESS_STATES | FAILURE_STATES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_state(job: Any) -> str:
    return getattr(job.state, "name", str(job.state))


def coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def format_duration(value: timedelta | None) -> str:
    if value is None:
        return "unknown duration"
    total_seconds = int(max(value.total_seconds(), 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes}:{seconds:02}"


def batch_elapsed(job: Any) -> timedelta | None:
    created = coerce_datetime(getattr(job, "create_time", None))
    ended = coerce_datetime(getattr(job, "end_time", None))
    updated = coerce_datetime(getattr(job, "update_time", None))
    finished = ended or updated
    if created is None or finished is None:
        return None
    return finished - created


def batch_age(job: Any) -> timedelta | None:
    created = coerce_datetime(getattr(job, "create_time", None))
    if created is None:
        return None
    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return now - created.astimezone(timezone.utc)


def summarize_error(error: Any) -> str | None:
    if error is None:
        return None
    message = getattr(error, "message", None)
    code = getattr(error, "code", None)
    if message:
        prefix = f"{code}: " if code is not None else ""
        return f"{prefix}{message}"
    if hasattr(error, "model_dump"):
        payload = error.model_dump(mode="json", exclude_none=True, by_alias=True)
        if isinstance(payload, dict):
            message = payload.get("message")
            code = payload.get("code")
            if message:
                prefix = f"{code}: " if code is not None else ""
                return f"{prefix}{message}"
            return json.dumps(payload, ensure_ascii=True)
    return str(error)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300
    except OSError as exc:
        print(f"notification_failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return False


def notify(title: str, message: str, bot_token_env: str, chat_id_env: str) -> bool:
    bot_token = os.getenv(bot_token_env, "").strip()
    chat_id = os.getenv(chat_id_env, "").strip()
    if not bot_token or not chat_id:
        return False
    return send_telegram_message(bot_token, chat_id, f"{title}\n\n{message}")


def ensure_state(state_path: Path, registry: dict[str, Any]) -> dict[str, Any]:
    state = load_json(state_path, default={"schema_version": 1, "jobs": {}})
    jobs_state = state.setdefault("jobs", {})
    for job in registry["jobs"]:
        jobs_state.setdefault(
            job["batch_job_name"],
            {
                "label": job["label"],
                "last_state": None,
                "last_polled_at": None,
                "terminal_notified_at": None,
            },
        )
    save_json(state_path, state)
    return state


def poll_once(
    client: Client,
    registry: dict[str, Any],
    state: dict[str, Any],
    send_notifications: bool,
    bot_token_env: str,
    chat_id_env: str,
) -> tuple[list[str], bool]:
    lines: list[str] = []
    all_terminal = True
    jobs_state = state["jobs"]

    for job_entry in registry["jobs"]:
        job_name = job_entry["batch_job_name"]
        batch_job: Any = client.batches.get(name=job_name)
        state_name = normalize_state(batch_job)
        error_summary = summarize_error(batch_job.error)
        elapsed = batch_elapsed(batch_job)
        age = batch_age(batch_job)
        if state_name in TERMINAL_STATES:
            lines.append(
                f"{job_entry['label']}: {state_name}  {job_name}  finished after {format_duration(elapsed)}"
            )
            if error_summary is not None:
                lines.append(f"  error: {error_summary}")
        else:
            lines.append(
                f"{job_entry['label']}: {state_name}  {job_name}  submitted {format_duration(age)} ago"
            )

        job_state = jobs_state[job_name]
        job_state["last_state"] = state_name
        job_state["last_polled_at"] = utc_now()

        if state_name not in TERMINAL_STATES:
            all_terminal = False
            continue

        if job_state["terminal_notified_at"] is None and send_notifications:
            if state_name in SUCCESS_STATES:
                title = f"MASK batch ready: {job_entry['label']}"
                message = (
                    f"{state_name} after {format_duration(elapsed)}\n"
                    f"{job_name}\nartifact={job_entry['artifact_dir']}"
                )
            else:
                title = f"MASK batch failed: {job_entry['label']}"
                message = (
                    f"{state_name} after {format_duration(elapsed)}\n"
                    f"{job_name}\nartifact={job_entry['artifact_dir']}"
                )
                if error_summary is not None:
                    message += f"\nerror={error_summary}"
            if notify(title, message, bot_token_env, chat_id_env):
                job_state["terminal_notified_at"] = utc_now()

    return lines, all_terminal


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Google batch jobs and optionally notify on terminal completion.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--telegram-bot-token-env", default=DEFAULT_BOT_ENV)
    parser.add_argument("--telegram-chat-id-env", default=DEFAULT_CHAT_ENV)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--exit-when-all-terminal", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    state_path = Path(args.state).resolve()

    registry = load_json(registry_path, default=None)
    if not registry or not registry.get("jobs"):
        raise SystemExit(f"No jobs found in registry: {registry_path}")

    client = Client(api_key=os.getenv("GOOGLE_API_KEY", "DUMMY"))
    state = ensure_state(state_path, registry)

    while True:
        lines, all_terminal = poll_once(
            client=client,
            registry=registry,
            state=state,
            send_notifications=not args.no_notify,
            bot_token_env=args.telegram_bot_token_env,
            chat_id_env=args.telegram_chat_id_env,
        )
        save_json(state_path, state)
        print(f"[{datetime.now().isoformat(timespec='seconds')}]")
        for line in lines:
            print(line)
        sys.stdout.flush()

        if args.once:
            return 0
        if args.exit_when_all_terminal and all_terminal:
            print("All tracked batch jobs are in terminal states.")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
