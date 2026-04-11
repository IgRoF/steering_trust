# Minimum eval

This is the shortest reliable path from a fresh local copy of the package to one successful MASK smoke run.

## Before you start

You need:

- Windows with PowerShell
- Python 3.12 (verify with `py -3.12 --version`)
- One provider API key
- A Hugging Face token with access to the public [MASK dataset](https://huggingface.co/datasets/cais/MASK)

If you are new to API-based model work, read [Provider setup and API keys](provider-setup-and-api-keys.md) first.

This package is validated on Python 3.12 because the pinned `inspect_evals` commit and the Google batch replay helper were both tested on that version. The current Inspect ecosystem usually supports Python 3.11 or 3.12, but you should still check the latest [Inspect docs](https://inspect.aisi.org.uk/), the [Inspect Evals package page](https://pypi.org/project/inspect-evals/), and the [MASK task page](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/mask/) before moving to a newer interpreter.

## What this smoke reproduces

A standard smoke run in this package uses:

- `inspect eval inspect_evals/mask`
- The public 1,000-example [MASK dataset from Hugging Face](https://huggingface.co/datasets/cais/MASK)
- 10 samples by default
- The same judge stack used in the main runs (`openai/gpt-4o` for binary, `openai/o3-mini` for numeric)

## Step 1: Create a virtual environment

Use the Python Launcher to make sure you create the virtual environment with Python 3.12. If your machine has multiple Python versions, running `python -m venv` might pick the wrong one.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

After the virtual environment is created, `.\.venv\Scripts\python` always points to the interpreter inside it. The same setup is described in [Environment and dependencies](../reference/environment-and-dependencies.md).

## Step 2: Check the environment

```powershell
.\.venv\Scripts\python .\scripts\check_environment.py
```

This prints the Python version, the installed package versions, and whether the package is still on the validated dependency baseline.

## Step 3: Set the required environment variables

For an OpenAI smoke, you need at least:

```powershell
$env:OPENAI_API_KEY = "your-key-here"
$env:HF_TOKEN = "your-hugging-face-token"
```

If your tools already use `HUGGINGFACE_TOKEN`, that is fine too. The wrapper scripts mirror the Hugging Face variable names when one of them is set.

## Step 4: Run a 10-sample smoke

The public MASK wrapper uses one script for both main modes. `-Mode smoke` selects the lighter 10-sample defaults. `-Mode full` selects the 1,000-sample defaults.

```powershell
.\scripts\run_mask.ps1 -Model "openai/gpt-4o-2024-08-06" -Mode smoke
```

The wrapper creates a dated run folder under `outputs/runs/`, prints the exact command it is about to run, launches `inspect_evals/mask` with the standard smoke defaults, and checks that the required environment variables are present.

## Step 5: Inspect the output

A successful smoke test gives you:

- A new run folder under `outputs/runs/`
- A `console.txt` file
- A `.eval` file
- A final score summary printed in the console

If you want a visual example first, see the [example smoke output bundle](../../examples/smoke-success/README.md).

If you want the full browser viewer instead of a text summary, read [Inspect log viewer](inspect-log-viewer.md).

To inspect one sample from a `.eval` file:

```powershell
.\.venv\Scripts\python .\scripts\show_eval_sample.py .\outputs\runs\<RUN_DIR>\<FILE>.eval
```

To summarize a real `.eval` file with `inspect_ai`:

```powershell
.\.venv\Scripts\python .\scripts\summarize_eval_log.py .\outputs\runs\<RUN_DIR>\<FILE>.eval
```

## When to move on to a full run

Run a full 1,000-sample eval only after:

1. Your smoke test works cleanly
2. Your provider setup is stable
3. You have checked [Cost estimation](cost-estimation.md)
4. You have checked [Rate limits and runtime](rate-limits-and-runtime.md)

Full-run example:

```powershell
.\scripts\run_mask.ps1 -Model "openai/gpt-5.4" -Mode full
```

You can also keep one mode and override the defaults yourself. For example, this keeps smoke test mode but raises the sample count:

```powershell
.\scripts\run_mask.ps1 -Model "openai/gpt-4o-2024-08-06" -Mode smoke -Limit 25
```

## Google batch alternative

If you want the Google batch-replay path:

```powershell
.\scripts\run_google_batch.ps1 -BatchModel "gemini-2.5-flash" -InspectModel "google/gemini-2.5-flash"
```

Read [Rate limits and runtime](rate-limits-and-runtime.md), [Optional monitoring](monitoring-optional.md), and [Telegram bot setup](telegram-bot-setup.md) before using that path.

## Common first failures

- `inspect.exe` not found: Recreate the virtual environment and keep the default wrapper path.
- Hugging Face access error: Check `HF_TOKEN` or `HUGGINGFACE_TOKEN`.
- Provider authentication error: Check that the matching `*_API_KEY` variable is set in the same shell.
- Quota or timeout error: Lower concurrency and read [Rate limits and runtime](rate-limits-and-runtime.md).
