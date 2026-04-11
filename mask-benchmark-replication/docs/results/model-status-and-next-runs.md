# Model status and next runs

This page keeps unfinished work visible without mixing it into the stable full-results table.

## Current status

| Model | `inspect_evals` model string | Provider model ref | Current state | Last run | Next step |
| --- | --- | --- | --- | --- | --- |
| GPT-4o overlap rerun | `openai/gpt-4o-2024-08-06` | gpt-4o-2024-08-06 | Completed full run | 2026-04-02 | Keep as the overlap anchor |
| DeepSeek V3.2 | `openai-api/deepseek/deepseek-chat` | DeepSeek-V3.2 | Completed full run | 2026-04-07 | Keep in the main comparison table |
| Grok 4.20 non-reasoning | `grok/grok-4.20-0309-non-reasoning` | grok-4.20-0309-non-reasoning | Completed full run | 2026-04-07 | Keep in the main comparison table |
| GPT-5.4 | `openai/gpt-5.4` | gpt-5.4 | Completed full run | 2026-04-07 | Keep in the main comparison table |
| Claude Opus 4.6 | `anthropic/claude-opus-4-6` | claude-opus-4-6 | Completed full run | 2026-04-07 | Keep in the main comparison table |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | gemini-2.5-flash | Completed full run | 2026-04-08 | Keep in the main comparison table |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` | gemini-3-flash-preview | Smoke only | 2026-04-08 | Decide later whether a full run is worth the extra reasoning-token cost |
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` | gemini-3.1-pro-preview | Smoke only | 2026-04-09 | Decide whether a full run is worth the observed target-side cost and reasoning-token use |
| Llama 3.1 8B OpenRouter rerun | `openrouter/meta-llama/llama-3.1-8b-instruct` | llama-3.1-8b-instruct | Completed full run with caveats | 2026-04-10 | Keep as the current provider-API Llama result |
| Llama 4 Scout | `openrouter/meta-llama/llama-4-scout` | llama-4-scout-17b-16e-instruct | Completed full run | 2026-04-10 | Keep in the main comparison table |
| Qwen 235B | `openrouter/qwen/qwen3-235b-a22b-2507` | qwen3-235b-a22b-2507 | Completed full run | 2026-04-10 | Keep in the main comparison table |

## Reading note

This page will change as new full runs arrive. The stable benchmark tables remain [Results overview](results-overview.md) and [Full runs](results-full-runs.md).
