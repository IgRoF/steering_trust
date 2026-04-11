# Inspect log viewer

This guide explains how to open a MASK `.eval` log in the Inspect browser viewer and how to read the main panels once it loads.

Official references:

- [Inspect log viewer guide](https://inspect.aisi.org.uk/log-viewer.html)
- [Inspect `view` command reference](https://inspect.aisi.org.uk/reference/inspect_view.html)
- [Inspect home and docs index](https://inspect.aisi.org.uk/)

## What the viewer is for

Use the Inspect viewer when you want more than a text summary. It is the easiest way to:

- Browse individual samples
- See the prompt, model response, and scores together
- Check run metadata such as model string, sample count, and token usage
- Filter or inspect failures, retries, and unusual samples

## Before you open a log

This package is validated on Python 3.12. That is the version used for the pinned `inspect_evals` commit and the public wrapper scripts.

Create the local environment first:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If you plan to use a different Python version, check the latest [Inspect docs](https://inspect.aisi.org.uk/) and [Inspect Evals package page](https://pypi.org/project/inspect-evals/) first.

## Open the bundled example log

From the package root:

```powershell
.\.venv\Scripts\inspect.exe view start --log-dir .\examples\smoke-success
```

Then open:

- `http://127.0.0.1:7575`

If port `7575` is already in use:

```powershell
.\.venv\Scripts\inspect.exe view start --log-dir .\examples\smoke-success --port 7576
```

Then open `http://127.0.0.1:7576`.

## Open one of your own runs

Point the viewer at the directory that contains the `.eval` file. For example:

```powershell
.\.venv\Scripts\inspect.exe view start --log-dir .\outputs\runs\<RUN_DIR>
```

The viewer reads the log directory, not just the single file path.

## What to look at inside the viewer

### Run metadata

Check the top-level run information first:

- Model string
- Task name
- Status
- Sample count
- Token usage, if present

This is the fastest way to confirm you opened the run you meant to inspect.

### Sample list

The sample list is where you move through the run one example at a time.

Useful patterns:

- Scan for errors or retries first
- Compare hard examples across similar runs
- Look for `provided_facts`, `disinformation`, or `continuations` examples if you want the most revealing MASK slices

### Messages and transcript

This view shows the actual prompt-response history for one sample. It helps answer:

- What pressure prompt was shown?
- What belief-elicitation answer did the model give?
- What pressured answer did the model give?
- Did the judges have enough information to classify the sample cleanly?

### Scores and metadata

This is where you check how the sample contributed to the benchmark metrics.

For MASK, the main questions are:

- Was the sample classified as honest, lie, evade, or no-belief?
- Did the sample count toward accuracy?
- Is this sample part of the `provided_facts` subset, where belief-based accuracy is treated differently?

If you are reading full-run results, keep the [Accuracy comparability](../results/accuracy-comparability.md) note nearby. Older runs and post-fix runs do not always use the same effective denominator.

## How to interpret the result at run level

The viewer helps you answer three different kinds of questions:

- **Did the run succeed operationally?**
  Look for status, sample count, errors, and retry patterns.
- **Did the model stay honest under pressure?**
  Look at the mix of honest, lie, evade, and no-belief outcomes.
- **Is the run comparable to the paper or to another run?**
  Check the run date, evaluator version, model string, and the relevant note in [Accuracy comparability](../results/accuracy-comparability.md).

## Public example vs. full private logs

The bundled example log is sanitized and small on purpose. It is there so you can learn the viewer without needing access to a full benchmark run.

The full raw `.eval` logs are not published in this package because the MASK benchmark still relies on a gated dataset and access requires permission from the authors. If you already have that permission and need the full logs for verification, contact me and I can share them privately on a case-by-case basis.

## Read next

- [Minimum eval](minimum-eval.md)
- [Results overview](../results/results-overview.md)
- [Accuracy comparability](../results/accuracy-comparability.md)
