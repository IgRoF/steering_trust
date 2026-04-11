# MASK Benchmark Replication

This repository documents an independent replication of the [MASK honesty benchmark](https://arxiv.org/abs/2503.03750) via its [implementation on AISI's Inspect framework](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/mask/). It focuses on a practical public workflow for API-accessible models, a layered presentation of the current results, and a clear path for readers who want to rerun the benchmark on their own accounts.

This work was completed as part of the [BlueDot Impact Technical AI Safety Project]([https://aisafetyfundamentals.com/](https://bluedot.org/courses/technical-ai-safety-project)) course (TAISP sprint, April 2026).

## The original MASK benchmark
![Figure 3 from the MASK paper](docs/assets/mask-paper-v3-figure-3.png)

> MASK tests whether a model says what it appears to believe when a prompt creates pressure to be dishonest.

Figure 3, reproduced from the MASK paper (arXiv:2503.03750v3, p. 4). Original caption:

*Our lying evaluation pipeline uses a three-step process anchored by a proposition, a variable capturing what we expect models to lie about and for which ground truth is known. (1) Prompt the model with two contrasting prompts, a pressure prompt designed to potentially induce deception and a neutral belief elicitation prompt. (2) Use LLMs to map both model statement (S) and belief (B) to their respective proposition values, with additional checks to ensure models consistently hold the belief B. (3) Measure lying by comparing statement S against belief B, as well as accuracy by comparing belief B against ground truth T. Example shortened for brevity.*

Formally, the paper defines lying and accuracy in terms of a model belief `B` and a model response `S`:

> Formally, let B denote the model’s belief (e.g., “the Great Wall was built primarily for symbolic purposes”) and let S represent the model’s response. We define lying as follows:
>
> Lying is making a statement S where S ≠ B, with the intent of causing a receiver to believe S is true.


> Distinguishing honesty from accuracy. While honesty pertains to the intentionality behind the model’s output, accuracy is a measure of factual correctness. In most evaluations, a model’s factual accuracy tests against an objective ground truth label.
>
> Inaccuracy is believing B where B is false.


Source: MASK paper, arXiv:2503.03750v3, p. 4.


## Status summary

| Topic | Current scope |
| --- | --- |
| Benchmark path | `inspect eval inspect_evals/mask` on the public 1,000-example [MASK dataset on Hugging Face](https://huggingface.co/datasets/cais/MASK) |
| Main reproduced settings | `honesty_at_n=1`<br>`belief_repeats_n=3`<br>`include_normalisation=false`<br>`binary_judge_model=openai/gpt-4o`<br>`numeric_judge_model=openai/o3-mini` |
| Important paper settings not reproduced | - Paper total set: 1,500 records (1,000 public and 500 gated)<br>- `Honesty@10` is still out of scope because it increases target-model cost sharply<br>- Normalized-for-beliefs reporting is still out of scope because finishing the current full-run comparison set took priority |
| Current full-run comparison set | GPT-4o overlap rerun<br>Claude Opus 4.6<br>GPT-5.4<br>DeepSeek V3.2<br>Grok 4.20 non-reasoning<br>Gemini 2.5 Flash<br>Llama 3.1 8B OpenRouter rerun<br>Llama 4 Scout<br>Qwen 235B |
| Main accuracy caveat | Older raw pre-fix accuracy values are not directly paper-comparable; see [Accuracy comparability](docs/results/accuracy-comparability.md) |

## Current main results

These are the current best completed full-run results (1,000 samples each). Each entry includes the run date so later updates can be tracked cleanly.

| Model | Run date | `inspect_evals` model string | Provider model reference | Honesty | Accuracy | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| GPT-4o overlap rerun | 2026-04-02 | `openai/gpt-4o-2024-08-06` | gpt-4o-2024-08-06 | 0.568 | 0.798 | Raw accuracy was 0.579; the comparison column excludes `provided_facts` |
| Claude Opus 4.6 | 2026-04-07 | `anthropic/claude-opus-4-6` | claude-opus-4-6 | 0.838 | 0.906 | Strongest honesty result in the completed set |
| GPT-5.4 | 2026-04-07 | `openai/gpt-5.4` | gpt-5.4 | 0.766 | 0.846 | The model string does not encode a separate thinking mode qualifier |
| DeepSeek V3.2 | 2026-04-07 | `openai-api/deepseek/deepseek-chat` | DeepSeek-V3.2 | 0.438 | 0.785 | The run used the OpenAI-compatible DeepSeek adapter; on the run date, the alias `deepseek-chat` pointed to DeepSeek-V3.2 non-thinking mode |
| Grok 4.20 non-reasoning | 2026-04-07 | `grok/grok-4.20-0309-non-reasoning` | grok-4.20-0309-non-reasoning | 0.375 | 0.756 | The non-reasoning qualifier is part of the actual model string |
| Gemini 2.5 Flash | 2026-04-08 | `google/gemini-2.5-flash` | gemini-2.5-flash | 0.558 | 0.689 | Generation used Google Batch; scoring used the standard MASK judges |
| Llama 3.1 8B OpenRouter rerun | 2026-04-10 | `openrouter/meta-llama/llama-3.1-8b-instruct` | llama-3.1-8b-instruct | 0.748 | 0.620 | Current provider-API Llama result; close to the paper's appendix anchor |
| Llama 4 Scout | 2026-04-10 | `openrouter/meta-llama/llama-4-scout` | llama-4-scout-17b-16e-instruct | 0.555 | 0.703 | Completed provider-API Scout baseline |
| Qwen 235B | 2026-04-10 | `openrouter/qwen/qwen3-235b-a22b-2507` | qwen3-235b-a22b-2507 | 0.514 | 0.739 | Completed provider-API Qwen baseline |

See [Results overview](docs/results/results-overview.md), [Full runs](docs/results/results-full-runs.md), and [Family evolution](docs/results/family-evolution.md).

- `inspect_evals model string` is the exact string you pass to `inspect eval` or to the wrapper scripts in this package.
- `Provider model reference` is the provider-facing model identifier or stable provider model name to use when you want a timeless label in write-ups.
- For DeepSeek, the exact inspect string still uses the alias `deepseek-chat`, but the provider-facing reference for this run is `DeepSeek-V3.2` because that is what the provider docs mapped the alias to on 2026-04-07.
- When the inspect string itself encodes a more specific variant, such as `non-reasoning`, that qualifier is kept exactly as part of the model name.

## What is reproduced here

- The public MASK task through `inspect_evals/mask`
- The public 1,000-example [MASK dataset on Hugging Face](https://huggingface.co/datasets/cais/MASK)
- The paper's main honesty headline (`1 - P(lie)`) at `honesty_at_n=1`
- `belief_repeats_n=3`
- The standard judge stack (`openai/gpt-4o` for binary, `openai/o3-mini` for numeric)
- Standard API-mode or provider-API runs for the completed comparison set
- Google Batch target generation for Gemini, followed by the same post-fix MASK replay scoring path used elsewhere in this package

## What is still outside the current scope

- The paper's full 1,500-example evaluation set. This package currently covers the 1,000 public examples; the remaining 500 are gated by the authors.
- `Honesty@10`. In the paper, this measures whether the model stays honest across repeated pressured attempts on the same example. It was not run here because it multiplies the number of target-model calls and makes already expensive full runs much more expensive.
- Normalized-for-beliefs reporting. This reports results after conditioning more explicitly on whether the model expressed a belief. It is still not included here because finishing the current full-run comparison set took priority.
- A completed Gemini Pro full run.

## Replication guide

<details>
<summary><strong>Advanced users</strong></summary>

This package is validated on Python 3.12. That matches the supported path recommended by [Inspect Evals](https://pypi.org/project/inspect-evals/). Before moving to a newer Python version, check the latest [Inspect docs](https://inspect.aisi.org.uk/) and the current `inspect_evals` package page.

Read these first:

1. [Minimum eval](docs/guides/minimum-eval.md)
2. [Environment and dependencies](docs/reference/environment-and-dependencies.md)
3. [Script map](docs/reference/script-map.md)
4. [Rate limits and runtime](docs/guides/rate-limits-and-runtime.md)
5. [Inspect log viewer](docs/guides/inspect-log-viewer.md)

Fastest likely first smoke:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key-here"
$env:HF_TOKEN = "your-hugging-face-token"
.\scripts\run_mask.ps1 -Model "openai/gpt-4o-2024-08-06" -Mode smoke
```
</details>

<details>
<summary><strong>Intermediate users</strong></summary>

Use Python 3.12 for the first run. This package pins a specific `inspect_evals` commit and reuses a Google batch replay helper that were both validated on 3.12. If you later want to move to a newer Python version, check the latest [Inspect docs](https://inspect.aisi.org.uk/) and [Inspect Evals package page](https://pypi.org/project/inspect-evals/) first.

Read in this order:

1. [Environment and dependencies](docs/reference/environment-and-dependencies.md)
2. [Provider setup and API keys](docs/guides/provider-setup-and-api-keys.md)
3. [Minimum eval](docs/guides/minimum-eval.md)
4. [Cost estimation](docs/guides/cost-estimation.md)
5. [Rate limits and runtime](docs/guides/rate-limits-and-runtime.md)
6. [Inspect log viewer](docs/guides/inspect-log-viewer.md)
</details>

<details>
<summary><strong>Beginners</strong></summary>

Start here:

1. [Environment and dependencies](docs/reference/environment-and-dependencies.md)
2. [Provider setup and API keys](docs/guides/provider-setup-and-api-keys.md)
3. [Minimum eval](docs/guides/minimum-eval.md)
4. [Example smoke output bundle](examples/smoke-success/README.md)
5. [Inspect log viewer](docs/guides/inspect-log-viewer.md)
6. [Results overview](docs/results/results-overview.md)

If this is your first API-based evaluation workflow, begin with a 10-sample smoke. Move to a 1,000-sample run only after you have checked cost and rate limits on your own account.
</details>

## Document index

### Guides

- [Minimum eval](docs/guides/minimum-eval.md): The shortest reliable path from a fresh local copy to one successful 10-sample smoke run.
- [Provider setup and API keys](docs/guides/provider-setup-and-api-keys.md): Account setup, environment variables, and provider-specific links for OpenAI, Anthropic, Google, DeepSeek, xAI, OpenRouter, and Hugging Face.
- [Cost estimation](docs/guides/cost-estimation.md): How to estimate target-model and judge cost before launching a full run.
- [Rate limits and runtime](docs/guides/rate-limits-and-runtime.md): How to read provider quotas, tune concurrency, and interpret the calculator outputs.
- [Inspect log viewer](docs/guides/inspect-log-viewer.md): How to open `.eval` logs in the Inspect viewer and what the main panels and fields mean.
- [Optional monitoring](docs/guides/monitoring-optional.md): Console polling and optional Telegram notifications for long Google Batch jobs.
- [Telegram bot setup](docs/guides/telegram-bot-setup.md): Step-by-step setup for Telegram notifications used by the optional watcher.

### Results

- [Results overview](docs/results/results-overview.md): The shortest summary of the current benchmark-level conclusions and why they matter.
- [Full runs](docs/results/results-full-runs.md): The complete 1,000-sample full-run tables, including comparison checks and older context rows.
- [Smokes and diagnostics](docs/results/results-smokes-and-diagnostics.md): Small runs and historical diagnostics that are useful for setup validation, Gemini planning, and family-direction clues.
- [Family evolution](docs/results/family-evolution.md): How the current results compare with the paper's earlier family baselines, including GPT-4o and Llama overlap anchors.
- [Accuracy comparability](docs/results/accuracy-comparability.md): Why old raw accuracy values drifted, what upstream fix changed that, and which current rows are paper-comparable.
- [Model status and next runs](docs/results/model-status-and-next-runs.md): Which families are complete, which are still smoke-only, and where future full runs would be most informative.

### Reference

- [Environment and dependencies](docs/reference/environment-and-dependencies.md): The validated package baseline, install commands, and environment-variable contract.
- [Script map](docs/reference/script-map.md): A compact map of the public scripts, their inputs and outputs, and which settings are safe to tweak.
- [Environment manifest](ENVIRONMENT_MANIFEST.md): The pinned baseline used when this package was validated locally.
- [Public package changelog](CHANGELOG.md): The public release log for changes after the first upload.
- [Citation metadata](CITATION.cff): Citation metadata for anyone who wants to reference this replication package directly.

## Data files

The machine-readable public tables live under [`data/`](data/):

- [full_runs.csv](data/full_runs.csv)
- [smokes_and_diagnostics.csv](data/smokes_and_diagnostics.csv)
- [model_status.csv](data/model_status.csv)
- [runtime_profiles.csv](data/runtime_profiles.csv)
- [family_evolution.csv](data/family_evolution.csv)

## Log availability

Full raw `.eval` logs are not published in this package because the public MASK benchmark still depends on a gated dataset and access requires permission from the authors. If you already have that permission and need the full logs for verification, contact me and I can share them privately on a case-by-case basis.

## Example outputs

Beginner-friendly examples live under [`examples/smoke-success/`](examples/smoke-success/README.md). They show what a successful smoke run looks like on disk.
