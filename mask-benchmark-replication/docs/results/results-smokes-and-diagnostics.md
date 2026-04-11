# Smokes and diagnostics

These results are useful for setup validation, cost planning, or family-direction clues. They are too small or too path-specific to be treated as benchmark conclusions.

## Pilot runs

| Model | Run date | `inspect_evals` model string | Samples | Honesty | Raw accuracy | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GPT-4o pilot | 2026-03-18 | `openai/gpt-4o` | 10 | 0.800 | 0.400 | First end-to-end pilot |
| Claude Sonnet 4.6 pilot | 2026-03-20 | `anthropic/claude-sonnet-4-6` | 10 | 1.000 | 0.400 | Anthropic access check |
| GPT-5.4 pilot | 2026-03-20 | `openai/gpt-5.4` | 10 | 1.000 | 0.400 | Early pilot before full runs |
| Gemini 3.1 Pro Preview pilot | 2026-03-20 | `google/gemini-3.1-pro-preview` | 10 | 0.900 | 0.100 | Access validation only |
| DeepSeek V3.2 pilot | 2026-03-20 | `openai-api/deepseek/deepseek-chat` | 10 | 0.700 | 0.300 | Adapter-path pilot; inspect used the DeepSeek alias that later mapped to V3.2 |
| Grok 4.1 Fast Reasoning pilot | 2026-03-20 | `grok/grok-4-1-fast-reasoning` | 10 | 0.500 | 0.500 | Historical xAI pilot path |

## Smoke runs and historical diagnostics

| Model | Run date | `inspect_evals` model string | Samples | Honesty | Raw accuracy | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Gemini 2.5 Flash batch smoke | 2026-04-08 | `google/gemini-2.5-flash` | 10 | 0.700 | 0.833 | First successful batch-replay smoke |
| Gemini 3 Flash Preview batch smoke | 2026-04-08 | `google/gemini-3-flash-preview` | 20 | 0.600 | 0.917 | Medium-thinking smoke with substantial reasoning-token use |
| Gemini 3.1 Pro Preview batch smoke | 2026-04-09 | `google/gemini-3.1-pro-preview` | 20 | 0.550 | 0.917 | Medium-thinking latest-Pro smoke; estimated target batch cost $0.3720 |
| Gemini 3 Pro Preview shard 1 | 2026-04-07 | `google/gemini-3-pro-preview` | 5 | 0.600 | 1.000 | Historical live shard only |
| Gemini 3 Pro Preview shard 2 | 2026-04-07 | `google/gemini-3-pro-preview` | 5 | 0.800 | 0.750 | Historical live shard only |

## What can be learned from these results

- Use pilots to check whether a provider path is alive.
- Use smoke tests to estimate cost and rate limits.
- For Gemini-Pro work, prefer the completed `google/gemini-3.1-pro-preview` batch smoke over the older live `google/gemini-3-pro-preview` shards when estimating a future full-run budget.
- Use diagnostics to explain historical decisions, such as why the package moved to the Google Batch path for serious Gemini work.

If you want the current benchmark-level results, go back to [Results overview](results-overview.md) or [Full runs](results-full-runs.md).
