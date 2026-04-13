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

## Detailed honesty breakdown (main comparison models)

All N=1000. Honesty = 1 - P(lie). Wilson 95% CIs.

| Model | Honest | Lie | Evade | No-Belief | Error | Honesty | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Claude Opus 4.6 | 690 (69.0%) | 162 (16.2%) | 117 (11.7%) | 23 (2.3%) | 8 (0.8%) | 0.838 | [0.814, 0.860] |
| GPT-5.4 | 504 (50.4%) | 234 (23.4%) | 189 (18.9%) | 70 (7.0%) | 3 (0.3%) | 0.766 | [0.739, 0.791] |
| Llama 3.1 8B OpenRouter | 198 (19.8%) | 252 (25.2%) | 407 (40.7%) | 141 (14.1%) | 2 (0.2%) | 0.748 | [0.720, 0.774] |
| GPT-4o overlap rerun | 219 (21.9%) | 432 (43.2%) | 255 (25.5%) | 77 (7.7%) | 17 (1.7%) | 0.568 | [0.537, 0.598] |
| Gemini 2.5 Flash | 210 (21.0%) | 442 (44.2%) | 205 (20.5%) | 136 (13.6%) | 7 (0.7%) | 0.558 | [0.527, 0.589] |
| Llama 4 Scout | 226 (22.6%) | 445 (44.5%) | 171 (17.1%) | 151 (15.1%) | 7 (0.7%) | 0.555 | [0.524, 0.586] |
| Qwen 235B | 267 (26.7%) | 486 (48.6%) | 122 (12.2%) | 120 (12.0%) | 5 (0.5%) | 0.514 | [0.483, 0.545] |
| DeepSeek V3.2 | 193 (19.3%) | 562 (56.2%) | 137 (13.7%) | 104 (10.4%) | 4 (0.4%) | 0.438 | [0.408, 0.469] |
| Grok 4.20 non-reasoning | 180 (18.0%) | 625 (62.5%) | 55 (5.5%) | 136 (13.6%) | 4 (0.4%) | 0.375 | [0.346, 0.405] |

Key observations:

- Llama 3.1 8B OpenRouter achieves high honesty primarily through extreme evasion (40.7%), not explicit honest responses (19.8%).
- Grok 4.20 has the highest lie rate (62.5%) and the lowest evasion rate (5.5%).
- Claude Opus 4.6 is the only model where the majority of responses are explicit `honest` labels (69.0%).

## Detailed accuracy breakdown (main comparison models)

Accuracy denominator = 726 non-`provided_facts` samples. Wilson 95% CIs.

| Model | Correct | Incorrect | No-Belief | Error | Accuracy | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Claude Opus 4.6 | 658 | 25 (3.4%) | 38 (5.2%) | 5 | 0.906 | [0.883, 0.925] |
| GPT-5.4 | 614 | 21 (2.9%) | 88 (12.1%) | 3 | 0.846 | [0.818, 0.870] |
| GPT-4o overlap rerun | 579 | 27 (3.7%) | 104 (14.3%) | 16 | 0.798 | [0.767, 0.825] |
| DeepSeek V3.2 | 570 | 29 (4.0%) | 123 (16.9%) | 4 | 0.785 | [0.754, 0.813] |
| Grok 4.20 non-reasoning | 549 | 31 (4.3%) | 143 (19.7%) | 3 | 0.756 | [0.724, 0.786] |
| Qwen 235B | 534 | 58 (8.0%) | 131 (18.0%) | 3 | 0.736 | [0.702, 0.766] |
| Llama 4 Scout | 508 | 40 (5.5%) | 175 (24.1%) | 3 | 0.700 | [0.665, 0.732] |
| Gemini 2.5 Flash | 500 | 40 (5.5%) | 182 (25.1%) | 4 | 0.689 | [0.654, 0.721] |
| Llama 3.1 8B OpenRouter | 450 | 32 (4.4%) | 243 (33.5%) | 1 | 0.620 | [0.584, 0.654] |

## Knows-but-lies analysis

Gap = accuracy minus honesty. Positive = model knows the truth but lies under pressure. Conservative 95% CIs.

| Model | Accuracy | Honesty | Gap (pp) | 95% CI | Evade/Lie Ratio |
| --- | ---: | ---: | ---: | --- | ---: |
| Grok 4.20 | 0.756 | 0.375 | +38.1 | [+31.9, +44.0] | 0.09 |
| DeepSeek V3.2 | 0.785 | 0.438 | +34.7 | [+28.5, +40.5] | 0.24 |
| GPT-4o | 0.798 | 0.568 | +23.0 | [+16.9, +28.8] | 0.59 |
| Qwen 235B | 0.736 | 0.514 | +22.2 | [+15.7, +28.3] | 0.25 |
| Llama 4 Scout | 0.700 | 0.555 | +14.5 | [+7.9, +20.8] | 0.38 |
| Gemini 2.5 Flash | 0.689 | 0.558 | +13.1 | [+6.5, +19.4] | 0.46 |
| GPT-5.4 | 0.846 | 0.766 | +8.0 | [+2.7, +13.1] | 0.81 |
| Claude Opus 4.6 | 0.906 | 0.838 | +6.8 | [+2.3, +11.1] | 0.72 |
| Llama 3.1 8B | 0.620 | 0.748 | -12.8 | [-19.0, -6.6] | 1.62 |

All gaps except Llama 3.1 8B have CIs entirely above zero. Llama 3.1 8B is the only model that is more honest than accurate, because its high evasion suppresses the lie rate while its high no-belief rate suppresses accuracy.

## Per-config honesty (1 - P(lie) per archetype)

| Model | continuations | disinformation | dd_known_facts | known_facts | provided_facts | statistics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Opus 4.6 | 0.847 | 0.672 | 0.833 | 0.842 | 0.876 | 0.927 |
| GPT-5.4 | 0.744 | 0.848 | 0.650 | 0.856 | 0.686 | 0.875 |
| Llama 3.1 8B OpenRouter | 0.761 | 0.872 | 0.758 | 0.856 | 0.536 | 0.917 |
| GPT-4o overlap rerun | 0.460 | 0.304 | 0.567 | 0.842 | 0.482 | 0.760 |
| Gemini 2.5 Flash | 0.290 | 0.568 | 0.600 | 0.876 | 0.383 | 0.792 |
| Llama 4 Scout | 0.557 | 0.280 | 0.633 | 0.675 | 0.478 | 0.771 |
| Qwen 235B | 0.432 | 0.672 | 0.475 | 0.670 | 0.321 | 0.719 |
| DeepSeek V3.2 | 0.261 | 0.264 | 0.467 | 0.689 | 0.361 | 0.625 |
| Grok 4.20 | 0.358 | 0.392 | 0.342 | 0.536 | 0.201 | 0.573 |

## Per-config accuracy (correct / (correct + incorrect + no-belief) per archetype)

`provided_facts` excluded (no accuracy denominator).

| Model | continuations | disinformation | dd_known_facts | known_facts | statistics |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude Opus 4.6 | 0.955 | 0.944 | 0.900 | 0.914 | 0.802 |
| GPT-5.4 | 0.915 | 0.920 | 0.800 | 0.794 | 0.817 |
| GPT-4o overlap rerun | 0.908 | 0.909 | 0.731 | 0.728 | 0.822 |
| DeepSeek V3.2 | 0.898 | 0.944 | 0.633 | 0.660 | 0.870 |
| Grok 4.20 | 0.847 | 0.832 | 0.717 | 0.684 | 0.720 |
| Qwen 235B | 0.920 | 0.944 | 0.642 | 0.569 | 0.624 |
| Llama 4 Scout | 0.818 | 0.904 | 0.650 | 0.665 | 0.366 |
| Gemini 2.5 Flash | 0.881 | 0.864 | 0.550 | 0.531 | 0.652 |
| Llama 3.1 8B OpenRouter | 0.773 | 0.792 | 0.583 | 0.545 | 0.326 |

## Scale Labs MASK leaderboard comparison

Compared on 2026-04-13. Scale uses 500 private examples; this project uses 1,000 public examples. Scale reports only honesty. See the [accuracy comparability page](accuracy-comparability.md) for why cross-source comparisons are directional only.

| This project's model | Scale match | Scale honesty | This project's honesty | Delta |
| --- | --- | ---: | ---: | --- |
| Claude Opus 4.6 | claude-opus-4-6 (Non-Thinking) | 0.963 | 0.838 | +12.5pp (Scale higher) |
| GPT-4o | gpt 4o (November 2024) | 0.601 | 0.568 | +3.3pp |
| Qwen 235B | Qwen3-235B-A22B | 0.564 | 0.514 | +5.0pp |
| Gemini 2.5 Flash | Gemini 2.5 Flash Preview (May 2025) | 0.491 | 0.558 | -6.7pp |

Key differences: (a) private vs public examples, (b) possible scorer/judge pipeline differences, (c) inference configuration not disclosed by Scale. The Claude gap (+12.5pp) is the most notable and likely reflects the combined effect of all three.

## Confidence interval methodology

All confidence intervals in this package use Wilson score intervals at the 95% level (z=1.96). Wilson intervals are preferred over normal approximation (Wald) intervals because they have better coverage properties near 0 and 1, never produce out-of-range values, and are recommended for proportions with moderate sample sizes.

At N=1000 (honesty) and N=726 (accuracy), headline CI widths are roughly 3 percentage points. Per-config CIs are wider, especially for `statistics` (N=96) where widths can reach 19 percentage points.

## Earlier context and superseded full runs

| Model | Run date | `inspect_evals` model string | Honesty | Raw accuracy | Why it still matters |
| --- | --- | --- | ---: | ---: | --- |
| Claude Opus 4.6 pre-fix run | 2026-03-24 | `anthropic/claude-opus-4-6` | 0.861 | 0.557 | Shows how much the comparison changed after the post-fix rerun |
| GPT-5.4 pre-fix batch run | 2026-03-25 | `openai/gpt-5.4` | 0.806 | 0.503 | Shows the older batch-path result that is no longer the main comparison |
| Llama 3.1 8B original run | 2026-03-24 | `meta-llama/Llama-3.1-8B-Instruct` | 0.702 | 0.306 | Overlap anchor before the denominator fix; compare with care |
