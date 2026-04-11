# Environment manifest

This file records the exact environment that the public scripts were written and checked against.

## Validated baseline

| Item | Version or value |
| --- | --- |
| Operating system | Windows 11 / PowerShell workflow |
| Python | 3.12 |
| `inspect-ai` | 0.3.201 |
| `inspect-evals` | Git commit `0b1a737cf1d4dbd32235f47bf30bd731873dfc27` |
| `openai` | 2.30.0 |
| `anthropic` | 0.86.0 |
| `google-genai` | 1.68.0 |
| `xai-sdk` | 1.9.1 |
| `huggingface-hub` | 1.2.0 |

## Why the requirements stay pinned

Two parts of this package are sensitive to dependency drift:

- `inspect_evals/mask` changed materially during this project because of upstream denominator and scorer fixes.
- `scripts/run_google_batch.py` imports private `inspect_ai` provider helpers for the Google batch replay path.

Using newer package versions is risky unless the package is revalidated. The public scripts are written for the versions above.

## How to check your environment

Run:

```powershell
.\.venv\Scripts\python .\scripts\check_environment.py
```

If the report shows drift, reinstall from `requirements.txt` before debugging the run logic.
