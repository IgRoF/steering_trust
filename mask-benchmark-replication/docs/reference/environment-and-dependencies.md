# Environment and dependencies

This package is documented and tested as a Windows plus PowerShell workflow. Other environments can still work, but the copy-paste commands here assume Windows paths.

## Validated baseline

| Item | Version or value |
| --- | --- |
| Python | `3.12` |
| `inspect-ai` | `0.3.201` |
| `inspect-evals` | Git commit `0b1a737cf1d4dbd32235f47bf30bd731873dfc27` |
| `openai` | `2.30.0` |
| `anthropic` | `0.86.0` |
| `google-genai` | `1.68.0` |
| `xai-sdk` | `1.9.1` |
| `huggingface-hub` | `1.2.0` |

See also [ENVIRONMENT_MANIFEST.md](../../ENVIRONMENT_MANIFEST.md).

The package is intentionally pinned to Python 3.12. That is the version used when the current `inspect_evals` commit and the Google batch replay helper were validated. The broader Inspect ecosystem often supports Python 3.11 or 3.12, but you should always check the latest [Inspect docs](https://inspect.aisi.org.uk/), the [Inspect Evals package page](https://pypi.org/project/inspect-evals/), and the current [MASK task documentation](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/mask/) before upgrading.

## Why the requirements stay pinned

Two parts of the workflow are sensitive to dependency drift:

- `inspect_evals/mask` changed materially during this project because of upstream denominator and scorer fixes.
- `scripts/run_google_batch.py` imports private `inspect_ai` provider helpers for the Google batch replay path.

Moving straight to the latest package versions is risky unless the workflow is revalidated.

## Install

On Windows, use the Python Launcher to make sure you create the virtual environment with the right Python version. If your machine has multiple Python versions installed, running `python -m venv` might pick the wrong one (for example, Python 3.14 instead of 3.12).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The `py -3.12` command explicitly selects Python 3.12 through the Windows Python Launcher. After the virtual environment is created, `.\.venv\Scripts\python` always points to the interpreter inside it, so the remaining commands are version-safe.

If you only have Python 3.12 installed, `python -m venv .venv` also works.

## Check the environment

```powershell
.\.venv\Scripts\python .\scripts\check_environment.py
```

This reports:

- The current Python version
- The installed package versions
- Any drift from the validated baseline
- Whether `inspect.exe` is available at the default wrapper path

## Environment variables

Set only the variables you need for the providers you plan to use.

| Variable | Used for | When it is required | Related guide |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | GPT-4o, GPT-5.4, and the default MASK judges | Required for OpenAI target runs and for judge-based scoring in the standard scripts | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `ANTHROPIC_API_KEY` | Claude target runs | Required only for Anthropic target runs | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `GOOGLE_API_KEY` | Gemini live or batch target runs | Required only for Google target runs | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `DEEPSEEK_API_KEY` | DeepSeek target runs | Required only for DeepSeek target runs | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `XAI_API_KEY` | Grok target runs | Required only for xAI target runs | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `OPENROUTER_API_KEY` | OpenRouter-backed open-weight runs (Llama, Scout, Qwen) | Required for OpenRouter target runs | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `HF_TOKEN` | Access to the public [MASK dataset on Hugging Face](https://huggingface.co/datasets/cais/MASK) | Required for real MASK eval runs; not required for calculators, docs, or example browsing | [Provider setup](../guides/provider-setup-and-api-keys.md) |
| `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` | Optional watcher notifications | Required only if you want Telegram alerts | [Telegram bot setup](../guides/telegram-bot-setup.md) |

If your machine already uses `HUGGINGFACE_TOKEN`, that is fine too. The wrapper scripts mirror that name with `HF_TOKEN`.

## Outputs

The public scripts write to `outputs/` by default. That folder is already included in the package with a `.gitkeep` placeholder so local forks have an obvious place to write run artifacts.

The main subfolders are:

- `outputs/runs/` for standard `.eval` logs
- `outputs/google-batch/` for Google batch artifacts and watch state
- `outputs/inspect-appdata/` for local Inspect appdata redirection used by the replay-scoring path

These outputs are local run artifacts and are ignored by this package's `.gitignore`.

## Read next

- [Provider setup and API keys](../guides/provider-setup-and-api-keys.md)
- [Minimum eval](../guides/minimum-eval.md)
- [Rate limits and runtime](../guides/rate-limits-and-runtime.md)
- [Inspect log viewer](../guides/inspect-log-viewer.md)
