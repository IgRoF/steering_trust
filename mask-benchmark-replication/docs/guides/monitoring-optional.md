# Optional monitoring

Monitoring is optional. You do not need it for a simple smoke run.

Use it when:

- A Google batch job may take hours
- You want console polling while you step away
- You want Telegram notifications for terminal job states

## Console-only Google batch watch

If you submitted a Google batch through [`run_google_batch.ps1`](../../scripts/run_google_batch.ps1), the package writes a registry under `outputs/google-batch/registry.json`.

Watch continuously:

```powershell
.\scripts\run_google_batch_monitor.ps1
```

One-shot poll:

```powershell
.\scripts\run_google_batch_monitor.ps1 -Once
```

Exit automatically once everything in the registry is terminal:

```powershell
.\scripts\run_google_batch_monitor.ps1 -ExitWhenAllTerminal
```

## What the watcher stores

- `outputs/google-batch/registry.json`
- `outputs/google-batch/watch_state.json`

These are local state files. They are ignored by `.gitignore`.

## Telegram notifications

Telegram is optional too. If you want it, set the variables below in the same shell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
```

Then run the watcher normally. It sends one notification per batch job when the job reaches a terminal state.

If you need setup instructions, read [Telegram bot setup](telegram-bot-setup.md).

When the batch finishes and replay scoring writes a `.eval` log, use [Inspect log viewer](inspect-log-viewer.md) to inspect the result in the browser.
