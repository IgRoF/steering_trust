#!/usr/bin/env python3
"""Check whether the local Python environment matches the validated package baseline."""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import metadata
from pathlib import Path

EXPECTED_PYTHON_PREFIX = "3.12"
EXPECTED_PACKAGES = {
    "inspect-ai": "0.3.201",
    "openai": "2.30.0",
    "anthropic": "0.86.0",
    "google-genai": "1.68.0",
    "xai-sdk": "1.9.1",
    "huggingface-hub": "1.2.0",
}
EXPECTED_INSPECT_EVALS_COMMIT = "0b1a737cf1d4dbd32235f47bf30bd731873dfc27"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def inspect_evals_version() -> str | None:
    return package_version("inspect-evals") or package_version("inspect_evals")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local package environment.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if drift is detected")
    args = parser.parse_args()

    drift_detected = False
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"python_version={python_version}")
    if not python_version.startswith(EXPECTED_PYTHON_PREFIX):
        drift_detected = True
        print(f"python_status=drift expected_prefix={EXPECTED_PYTHON_PREFIX}")
    else:
        print("python_status=ok")

    for package_name, expected in EXPECTED_PACKAGES.items():
        actual = package_version(package_name)
        if actual is None:
            drift_detected = True
            print(f"package_status={package_name}:missing expected={expected}")
            continue
        status = "ok" if actual == expected else "drift"
        if status != "ok":
            drift_detected = True
        print(f"package_status={package_name}:{status} expected={expected} actual={actual}")

    inspect_evals_actual = inspect_evals_version()
    if inspect_evals_actual is None:
        drift_detected = True
        print(f"inspect_evals_status=missing expected_commit={EXPECTED_INSPECT_EVALS_COMMIT}")
    else:
        status = "ok" if EXPECTED_INSPECT_EVALS_COMMIT in inspect_evals_actual else "check"
        if status != "ok":
            drift_detected = True
        print(
            f"inspect_evals_status={status} expected_commit={EXPECTED_INSPECT_EVALS_COMMIT} actual={inspect_evals_actual}"
        )

    inspect_exe = project_root() / ".venv" / "Scripts" / "inspect.exe"
    print(f"inspect_exe_default={inspect_exe}")
    print(f"inspect_exe_exists={'yes' if inspect_exe.exists() else 'no'}")
    print(f"python_executable={sys.executable}")
    print(f"pip_path={shutil.which('pip') or 'not_on_path'}")

    if drift_detected and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
