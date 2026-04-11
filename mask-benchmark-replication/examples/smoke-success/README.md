# Example smoke output bundle

This folder shows what a successful Inspect `.eval` bundle looks like on disk.

It contains:

- `example_console.txt`: a short transcript from the helper script
- `example_summary.json`: a compact summary in JSON form
- `example_smoke.eval`: a sanitized `.eval` bundle based on a real Inspect log layout

## Quick checks

Inspect the example bundle:

```powershell
.\.venv\Scripts\python .\scripts\show_eval_sample.py .\examples\smoke-success\example_smoke.eval
```

Open the example in the Inspect log viewer:

```powershell
.\.venv\Scripts\inspect.exe view start --log-dir .\examples\smoke-success
```

If you are new to the viewer, read [Inspect log viewer](../../docs/guides/inspect-log-viewer.md).

Read the static summary:

```powershell
Get-Content .\examples\smoke-success\example_summary.json
```

## What the example bundle contains

The bundle contains 15 synthetic samples:

- 5 `known_facts` samples
- 5 `continuations` samples
- 5 `statistics` samples

Each archetype uses one base situation and then varies the model behavior across:

- `honest`
- `evade`
- `lie`
- `no-belief`
- `error`

Every sample includes:

- belief elicitation prompts and model answers
- one pressured prompt and answer
- synthetic judge prompts and judge answers
- the final `accuracy` and `honesty` score recorded in the log

The synthetic judge traces are stored under `store.synthetic_judge_records` inside each sample JSON.

## Why the example `.eval` is sanitized

The example bundle is meant for orientation. It helps readers recognize the file layout and the kind of information that appears in a real MASK run without exposing private logs or dataset content.

The `.eval` file is built from a real Inspect journal shape and then populated with public-safe synthetic examples. That keeps it compatible with log readers while avoiding local operational details and avoiding any direct reuse of gated dataset items.
