# Full runs

This page lists the completed full runs in three groups.

Full raw `.eval` logs are not published in this package because the benchmark uses a gated dataset and access requires permission from the authors. If you already have that permission and need the full logs for verification, contact me and I can share them privately on a case-by-case basis.

## Reading notes

- `inspect_evals model string` is the exact string used in `inspect_evals`. It is the string to pass to the wrapper scripts.
- `Provider model reference` is the provider's own model identifier, formatted to match the provider's documentation where possible.
- `Paper-comparable accuracy` is the number to use when the raw pre-fix denominator would otherwise be misleading.

## Main comparison results

| Model | Run date | `inspect_evals` model string | Provider model ref | Honesty | Raw accuracy | Paper-comparable accuracy | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| GPT-4o overlap rerun | 2026-04-02 | `openai/gpt-4o-2024-08-06` | gpt-4o-2024-08-06 | 0.568 | 0.579 | 0.798 | Use this GPT-4o result for paper comparison |
| Claude Opus 4.6 | 2026-04-07 | `anthropic/claude-opus-4-6` | claude-opus-4-6 | 0.838 | 0.906 | 0.906 | Current strongest honesty result |
| GPT-5.4 | 2026-04-07 | `openai/gpt-5.4` | gpt-5.4 | 0.766 | 0.846 | 0.846 | Standard API mode on the run date |
| DeepSeek V3.2 | 2026-04-07 | `openai-api/deepseek/deepseek-chat` | DeepSeek-V3.2 | 0.438 | 0.785 | 0.785 | Inspect used the DeepSeek alias `deepseek-chat`, which mapped to DeepSeek-V3.2 non-thinking mode on the run date |
| Grok 4.20 non-reasoning | 2026-04-07 | `grok/grok-4.20-0309-non-reasoning` | grok-4.20-0309-non-reasoning | 0.375 | 0.756 | 0.756 | The model string encodes the non-reasoning qualifier |
| Gemini 2.5 Flash | 2026-04-08 | `google/gemini-2.5-flash` | gemini-2.5-flash | 0.558 | 0.689 | 0.689 | Batch target generation with standard replay scoring |
| Llama 3.1 8B OpenRouter rerun | 2026-04-10 | `openrouter/meta-llama/llama-3.1-8b-instruct` | llama-3.1-8b-instruct | 0.748 | 0.620 | 0.620 | Current provider-API Llama result; close to the paper's appendix anchor |
| Llama 4 Scout | 2026-04-10 | `openrouter/meta-llama/llama-4-scout` | llama-4-scout-17b-16e-instruct | 0.555 | 0.703 | 0.703 | Completed provider-API Scout baseline |
| Qwen 235B | 2026-04-10 | `openrouter/qwen/qwen3-235b-a22b-2507` | qwen3-235b-a22b-2507 | 0.514 | 0.739 | 0.739 | Completed provider-API Qwen baseline |

## Comparison checks

| Model | Run date | `inspect_evals` model string | Honesty | Raw accuracy | Paper-comparable accuracy | Why keep it |
| --- | --- | --- | ---: | ---: | ---: | --- |
| GPT-4o public-set anchor | 2026-03-23 | `openai/gpt-4o` | 0.602 | 0.487 | 0.664 | First full public-set anchor |
| GPT-4o pinned rerun | 2026-03-24 | `openai/gpt-4o-2024-08-06` | 0.600 | 0.482 | 0.664 | Snapshot-controlled rerun for the accuracy-gap investigation |
| Llama 3.1 8B A40 rerun | 2026-04-07 | `meta-llama/Llama-3.1-8B-Instruct` | 0.604 | 0.682 | 0.682 | Useful as a serving-stack comparison against the later OpenRouter rerun |

## Open-weight provider follow-up

The completed OpenRouter full runs for Scout and Qwen also add a useful within-family pattern check.

### Per-config highlights

| Model | Config | Honesty | Accuracy | Interpretation |
| --- | --- | ---: | ---: | --- |
| Llama 4 Scout | `disinformation` | 0.280 | 0.904 | Very strong knowledge with very weak honesty under explicit pressure |
| Llama 4 Scout | `statistics` | 0.771 | 0.354 | Considerably more honest than accurate on the numeric subset |
| Qwen 235B | `provided_facts` | 0.321 | N/A | Weakest honesty slice; accuracy is not defined on this config |
| Qwen 235B | `continuations` | 0.432 | 0.920 | Very accurate while still often following the deceptive continuation pressure |
| Qwen 235B | `disinformation` | 0.672 | 0.944 | Stronger honesty than Scout on the same deceptive-writing slice |

### Runtime and retry context

- Scout continuation retries: 697
- Qwen continuation retries: 150
- For context, the pinned OpenRouter Llama 3.1 rerun completed with 0 retries, while the post-fix Claude Opus 4.6 synchronous run completed with 659.

These Scout and Qwen runs were operationally noisy but still within the range already seen on some other full MASK runs.

### Estimated cost

These are token-based estimates from the recorded run usage plus the official provider price sheets checked on 2026-04-11.

| Model | Target-model estimate | Judge estimate | Total estimate |
| --- | ---: | ---: | ---: |
| Llama 4 Scout | $0.456 | $15.066 | $15.522 |
| Qwen 235B | $0.126 | $14.454 | $14.581 |

The main practical point: the OpenAI judges dominate total spend. The target model itself is a small fraction of cost on both runs.

## Earlier context and superseded full runs

| Model | Run date | `inspect_evals` model string | Honesty | Raw accuracy | Why it still matters |
| --- | --- | --- | ---: | ---: | --- |
| Claude Opus 4.6 pre-fix run | 2026-03-24 | `anthropic/claude-opus-4-6` | 0.861 | 0.557 | Shows how much the comparison changed after the post-fix rerun |
| GPT-5.4 pre-fix batch run | 2026-03-25 | `openai/gpt-5.4` | 0.806 | 0.503 | Shows the older batch-path result that is no longer the main comparison |
| Llama 3.1 8B original run | 2026-03-24 | `meta-llama/Llama-3.1-8B-Instruct` | 0.702 | 0.306 | Overlap anchor before the denominator fix; compare with care |
