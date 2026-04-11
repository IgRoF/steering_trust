# Provider setup and API keys

This guide explains how to prepare the accounts and environment variables used by the package.

Environment variables are the standard setup path for this kind of API-based replication work. They are widely supported, easy to rotate, and stay out of committed files if you keep them in the current shell or in a local `.env` file. You can see the package template in [`.env.example`](../../.env.example).

## What each variable is for

| Variable | Needed for | Required? |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI targets and the default MASK judges | Required for OpenAI runs; also required for judge-based scoring in the standard scripts |
| `ANTHROPIC_API_KEY` | Claude runs | Required only for Anthropic target runs |
| `GOOGLE_API_KEY` | Gemini batch or live runs | Required only for Google target runs |
| `DEEPSEEK_API_KEY` | DeepSeek target runs | Required only for DeepSeek target runs |
| `XAI_API_KEY` | Grok target runs | Required only for xAI target runs |
| `OPENROUTER_API_KEY` | OpenRouter-backed open-weight runs (Llama, Scout, Qwen) | Required for OpenRouter target runs |
| `HF_TOKEN` | Access to the public [MASK dataset on Hugging Face](https://huggingface.co/datasets/cais/MASK) | Required for actual MASK eval runs; not required for calculators or static result browsing |
| `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` | Optional watcher notifications | Optional |

## OpenAI

1. Create or open your OpenAI API organization.
2. Create an API key in the [OpenAI API key dashboard](https://platform.openai.com/settings/organization/api-keys).
3. Check your current limits in the [OpenAI rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits) and in the limits page inside the dashboard.
4. Add billing if you want higher limits or longer runs.

```powershell
$env:OPENAI_API_KEY = "your-key-here"
```

Used for: `openai/gpt-4o-2024-08-06`, `openai/gpt-5.4`, and the default judges (`openai/gpt-4o` binary, `openai/o3-mini` numeric).

## Anthropic

1. Create an Anthropic account and read the [API overview](https://docs.anthropic.com/en/api/overview).
2. Generate an API key in the Anthropic Console.
3. Review the [Anthropic rate limits documentation](https://docs.anthropic.com/en/api/rate-limits).
4. If needed, set a workspace spend limit before large runs.

```powershell
$env:ANTHROPIC_API_KEY = "your-key-here"
```

Used for: `anthropic/claude-opus-4-6`, `anthropic/claude-sonnet-4-6`.

## Google

1. Open [Using Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key).
2. Create an API key in Google AI Studio.
3. Review [Gemini pricing](https://ai.google.dev/pricing) and [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
4. If you plan to use batch jobs, confirm that your tier supports the models you want.

```powershell
$env:GOOGLE_API_KEY = "your-key-here"
```

Used for: `google/gemini-2.5-flash`, `google/gemini-3-flash-preview`, `google/gemini-3.1-pro-preview`.

## DeepSeek

1. Create a DeepSeek account.
2. Generate an API key in the DeepSeek platform.
3. Review the [DeepSeek pricing page](https://api-docs.deepseek.com/quick_start/pricing-details-usd).
4. Review the [DeepSeek model updates page](https://api-docs.deepseek.com/updates).
5. For model identity, note that on the 2026-04-07 run date the provider alias `deepseek-chat` pointed to `DeepSeek-V3.2` non-thinking mode.

```powershell
$env:DEEPSEEK_API_KEY = "your-key-here"
```

Used for: `openai-api/deepseek/deepseek-chat`. The package uses the OpenAI-compatible adapter path for DeepSeek, so the inspect string is different from the provider's own model name. In the result tables, the stable provider-facing reference is `DeepSeek-V3.2`.

## xAI

1. Create an xAI team or project.
2. Create an API key in the xAI console.
3. Review [Models and pricing](https://docs.x.ai/developers/models?cluster=us-east-1) and [Consumption and rate limits](https://docs.x.ai/docs/key-information/consumption-and-rate-limits).
4. Check both your rate limits and your spend or credit state.

```powershell
$env:XAI_API_KEY = "your-key-here"
```

Used for: `grok/grok-4.20-0309-non-reasoning`.

## OpenRouter

1. Create an OpenRouter account at [openrouter.ai](https://openrouter.ai/).
2. Add at least $10 in credits. The [OpenRouter limits guide](https://openrouter.ai/docs/api/reference/limits) documents that accounts with $10+ in credits have no platform-level rate limits on paid models.
3. Still watch for provider-side throttling and Cloudflare protection. OpenRouter warns that reasonable-usage protections can still apply.

```powershell
$env:OPENROUTER_API_KEY = "your-key-here"
```

Used for: `openrouter/meta-llama/llama-3.1-8b-instruct`, `openrouter/meta-llama/llama-4-scout`, `openrouter/qwen/qwen3-235b-a22b-2507`.

## Hugging Face

The actual benchmark data comes from the public [MASK dataset on Hugging Face](https://huggingface.co/datasets/cais/MASK). Without a Hugging Face token, the package can still read the docs, run the offline calculators, and inspect existing local files. It cannot run `inspect_evals/mask` end to end.

```powershell
$env:HF_TOKEN = "your-token-here"
$env:HUGGINGFACE_TOKEN = $env:HF_TOKEN
```

The wrapper scripts already mirror these names when one is set.

## Telegram

Telegram is only needed for optional watcher notifications. See [Telegram bot setup](telegram-bot-setup.md) and [Optional monitoring](monitoring-optional.md).
