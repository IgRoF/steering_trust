# Script map

This package exposes a small public script surface. You do not need to browse any external script history to run the public workflow.

## Quick reference

| Script | Purpose | Main inputs | Main outputs | Where to tweak it |
| --- | --- | --- | --- | --- |
| [`run_mask.ps1`](../../scripts/run_mask.ps1) | Unified MASK runner for smoke and full modes | `inspect_evals` model string, provider key, `HF_TOKEN`, optional `-Mode` | `outputs/runs/<RUN_DIR>/` | `-Mode`, limit, concurrency, timeout, retries |
| [`run_google_batch.ps1`](../../scripts/run_google_batch.ps1) | PowerShell wrapper for the Google batch pipeline | Batch model name, inspect model string, `GOOGLE_API_KEY`, `HF_TOKEN` | `outputs/google-batch/` and scored logs under `outputs/runs/` | Batch mode, thinking settings, artifact paths |
| [`run_google_batch_monitor.ps1`](../../scripts/run_google_batch_monitor.ps1) | PowerShell wrapper for the Google batch watcher | Registry path, optional Telegram env vars | Console status plus optional Telegram messages | Poll interval, registry and state paths |
| [`run_google_batch.py`](../../scripts/run_google_batch.py) | Google batch generation and replay scoring engine | Live provider API, batch mode flags, exact model strings | Batch artifacts, registry entries, scored `.eval` logs | Batch request shape and artifact roots |
| [`watch_google_batch_jobs.py`](../../scripts/watch_google_batch_jobs.py) | Google batch watcher | `outputs/google-batch/registry.json`, optional Telegram env vars | Console status plus optional notifications | Poll interval and notification env names |
| [`check_environment.py`](../../scripts/check_environment.py) | Baseline environment check | Local Python environment and `requirements.txt` | Console report only | Baseline version table |
| [`estimate_mask_cost.py`](../../scripts/estimate_mask_cost.py) | Cost calculator | Manual token counts or one `run_profile_label` from `data/runtime_profiles.csv` | Console report only | Price assumptions and sample scaling |
| [`rate_limit_calculator.py`](../../scripts/rate_limit_calculator.py) | Runtime and concurrency calculator | Requests-per-sample, tokens-per-sample, provider limits | Console report only | Your own RPM and TPM inputs |
| [`summarize_eval_log.py`](../../scripts/summarize_eval_log.py) | JSON summary for a real `.eval` file | One `.eval` path readable by `inspect_ai` | JSON to stdout | Use [Inspect log viewer](../guides/inspect-log-viewer.md) when you want the browser UI instead |
| [`show_eval_sample.py`](../../scripts/show_eval_sample.py) | Quick sample inspection for a `.eval` file | One `.eval` path | Compact text summary to stdout | Use [Inspect log viewer](../guides/inspect-log-viewer.md) when you want the browser UI instead |

Only `estimate_mask_cost.py` reads a public CSV file (`data/runtime_profiles.csv`, when you pass `--run-profile-label` or `--profile`). The other scripts do not depend on the data files. The result CSVs are mainly for the documentation pages.

## Which scripts to use first

- Start with [`run_mask.ps1`](../../scripts/run_mask.ps1) in `-Mode smoke` if you want the shortest end-to-end check.
- Use [`estimate_mask_cost.py`](../../scripts/estimate_mask_cost.py) after the smoke if you want a first budget estimate.
- Use [`rate_limit_calculator.py`](../../scripts/rate_limit_calculator.py) before `run_mask.ps1 -Mode full` if you want a safer starting point for concurrency.
- Use [`run_google_batch.ps1`](../../scripts/run_google_batch.ps1) and [`run_google_batch_monitor.ps1`](../../scripts/run_google_batch_monitor.ps1) only when you are intentionally using the Google batch path.
