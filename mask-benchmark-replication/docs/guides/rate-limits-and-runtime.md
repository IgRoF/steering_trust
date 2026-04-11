# Rate limits and runtime

Full MASK runs are often limited by provider quotas rather than by your local machine.

## What to collect before a full run

For each provider model you plan to use, collect:

- Requests per minute (RPM)
- Tokens per minute (TPM)
- Requests per day (RPD), if the provider uses it
- Any spend cap or prepaid balance rule that can stop the run even when RPM and TPM still look generous

## Where to find limits for each provider

### OpenAI

1. Read the [OpenAI rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits).
2. Open your OpenAI dashboard and find the limits page for the organization that owns the key.
3. Record the RPM and TPM for the exact model family you plan to use.
4. Re-check after billing or tier changes; OpenAI limits can rise as account history grows.

### Anthropic

1. Read the [Anthropic rate limits page](https://docs.anthropic.com/en/api/rate-limits).
2. Open the Anthropic Console for the workspace that owns the key.
3. Record the limits for the model family you plan to use.
4. Check both rate limits and spend settings before long runs.

### Google

1. Read the [Gemini rate limits guide](https://ai.google.dev/gemini-api/docs/rate-limits).
2. Follow the "View your active rate limits in AI Studio" link from that page.
3. Record the interactive limits and, if relevant, the batch limits for the target model.
4. Note whether you are using live interactive requests or the separate batch path in this package.

### OpenRouter

1. Read the [OpenRouter pricing FAQ](https://openrouter.ai/pricing) and the [OpenRouter limits guide](https://openrouter.ai/docs/api/reference/limits).
2. For pay-as-you-go usage, note the documented rule that accounts with at least $10 in credits have no platform-level rate limits on paid models.
3. Still watch for provider-side throttling and Cloudflare protection, because OpenRouter explicitly warns that reasonable-usage protections can still apply.
4. Treat any observed clean high-concurrency run as route-specific evidence, not as a universal guarantee across all OpenRouter models.

### DeepSeek

1. Open your DeepSeek console and find the key or usage section.
2. Record any explicit RPM, TPM, or concurrency limits shown there.
3. Review [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing-details-usd) before large runs.
4. If the console does not expose a clear limit table, start with a smoke and conservative concurrency.

### xAI

1. Read [Consumption and rate limits](https://docs.x.ai/docs/key-information/consumption-and-rate-limits).
2. Open the xAI console and check both usage state and remaining credit or spend position.
3. Record any RPM, TPM, or credit rules relevant to the exact Grok model path you plan to use.
4. Treat spend exhaustion as a real run blocker, even if the nominal rate limits look fine.

See also [Provider setup and API keys](provider-setup-and-api-keys.md).

This package is validated on Python 3.12 because that is the interpreter used for the pinned `inspect_evals` commit and the Google batch replay helper. Before changing Python versions, check the latest [Inspect docs](https://inspect.aisi.org.uk/) and [Inspect Evals package page](https://pypi.org/project/inspect-evals/).

## Why MASK needs more planning than a simple prompt script

One MASK sample can require multiple target-model calls, binary-judge calls, and numeric-judge calls. That means the judge path can be the real bottleneck even when the target model looks fast.

## The practical settings in the PowerShell wrappers

Example full-run command:

```powershell
.\scripts\run_mask.ps1 `
  -Mode full `
  -Model "openai/gpt-5.4" `
  -MaxConnections 32 `
  -MaxSamples 64 `
  -Timeout 180 `
  -MaxRetries 15 `
  -RetryOnError 3 `
  -FailOnError 25 `
  -LogBuffer 1
```

### `-MaxConnections`

Maximum number of concurrent in-flight tasks. This is the main pressure knob for provider quotas. Raise it when the provider is clearly underused. Lower it when you see repeated 429 errors, timeouts, or retry storms.

### `-MaxSamples`

How many samples Inspect keeps scheduled at once. Higher values keep the run busier. Lower values are easier to debug and easier to recover after quota trouble.

### `-RetryOnError`

How many times to retry a sample-level failure before giving up on that sample. Useful for occasional transient failures. Too high a value can waste time and cost when the configuration is unhealthy.

### `-FailOnError`

Number of sample-level failures allowed before the entire run stops. A lower value fails fast. A higher value tolerates more turbulence, but can hide a broken setup.

### `-LogBuffer`

How often logs are flushed. A low value is easier to monitor in real time. A higher value can reduce log-write overhead.

### `-Timeout`

Per-request timeout in seconds. Raise it only when the provider is healthy but slow. Do not use a larger timeout to paper over quota problems.

### `-MaxRetries`

Transport-level retry budget for provider calls. Keep it moderate. Very high values can hide an unhealthy configuration and inflate cost.

### `-Mode`

`smoke` applies the lighter defaults for a first check (`Limit=10`, `MaxConnections=32`, `MaxSamples=32`). `full` applies the package defaults for a complete run (`Limit=1000`, `MaxConnections=128`, `MaxSamples=160`). You can still override any of those values directly.

## Rate-limit calculator

Use the calculator after a smoke run gives you rough request and token counts per sample.

Example:

```powershell
.\.venv\Scripts\python .\scripts\rate_limit_calculator.py `
  --samples 1000 `
  --target-requests-per-sample 4 `
  --target-tokens-per-sample 1800 `
  --target-rpm 600 `
  --target-tpm 800000 `
  --binary-judge-requests-per-sample 1 `
  --binary-judge-tokens-per-sample 3700 `
  --binary-judge-rpm 1200 `
  --binary-judge-tpm 3000000 `
  --numeric-judge-requests-per-sample 1 `
  --numeric-judge-tokens-per-sample 550 `
  --numeric-judge-rpm 1200 `
  --numeric-judge-tpm 1500000
```

### Reading the calculator output

- `target_samples_per_minute` is the rough throughput ceiling for the target model path.
- `target_limiter=requests` means RPM is tighter than TPM for that path. `target_limiter=tokens` means TPM is the tighter limit. The same interpretation applies to the binary and numeric judge lines.
- `estimated_min_runtime_minutes` is a rough lower bound from the slowest of the three paths.
- `recommended_max_connections` and `recommended_max_samples` are conservative starting points.

## A safe tuning sequence

1. Run a small smoke.
2. Estimate runtime with your real limits.
3. Launch a medium-sized run if you still want more evidence.
4. Launch the full 1,000-sample run only after the medium path looks healthy.

Observed examples from the completed runs:

- The April 10 Llama 3.1 OpenRouter continuation completed at `MaxConnections=64` and `MaxSamples=64` with 0 HTTP retries. Treat this as evidence for that exact account and model route, not as a universal default.
- The April 10 Scout OpenRouter continuation completed at `MaxConnections=128` and `MaxSamples=160` with 697 HTTP retries and no sustained rate-limit cascade.
- The April 10 Qwen OpenRouter continuation completed at `MaxConnections=128` and `MaxSamples=160` with 150 HTTP retries and no sustained rate-limit cascade.

## Google batch path

Google Batch has two separate time components: the provider-side batch generation wait, and the local replay scoring time after the batch finishes. Plan those separately. If you want notifications while you wait, read [Optional monitoring](monitoring-optional.md) and [Telegram bot setup](telegram-bot-setup.md).

### Why this package uses Google batch for Gemini models

Google's interactive API enforces requests-per-day (RPD) limits that can make a 1,000-sample MASK run impractical. The Gemini 3 Pro Preview interactive shards in the [smokes and diagnostics](../results/results-smokes-and-diagnostics.md) table illustrate the problem: only 5 samples per shard before hitting RPD. The Google Batch API avoids RPD limits entirely, which is why the Gemini 2.5 Flash full run and the later Gemini smokes all used batch generation.

### Why the Inspect Evals built-in batch mode is not recommended

The `inspect_evals` framework supports its own [batch API integration](https://inspect.aisi.org.uk/models-batch.html) for several models. In practice, this path has a retry problem for this benchmark: any samples that fail get sent in new batches, and each batch can take up to 24 hours to process. If multiple retry rounds happen, the total wall-clock time grows fast. Any interruption during this process can also leave unused batches that cannot be recycled later.

The package's own `run_google_batch.py` script handles this differently: it generates all batch requests up front, submits them as a single batch, waits for the result, and then replays the batch responses through the standard MASK scoring pipeline locally. This keeps the batch round-trips predictable.

The Google Batch approach may also work for other model providers that support a batch API, but this package has only tested the Google path.

## Signs that your settings are too aggressive

- Repeated 429 errors
- Repeated transport timeouts
- Long stretches with no sample progress
- Retry storms that raise cost without helping completion
- A growing provider bill with very little finished work

When that happens, lower concurrency first. Then reassess timeouts and retry counts.
