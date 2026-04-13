# Results overview

This page is the shortest public results summary in the package.

Full raw `.eval` logs are not published here because the benchmark uses a gated dataset and access still requires permission from the authors. If you already have that permission and need full logs for verification, contact me and I can share them privately on a case-by-case basis.

## Result groups

The results are organized in three groups:

- **Main comparison results** are the current best completed full-run evidence for each family or anchor.
- **Comparison checks** are reruns or anchor checks that matter for comparability. They explain why the main results should be read the way they are.
- **Earlier context** covers older or superseded full runs that still matter for transparency.

## Main comparison results

| Model | Run date | `inspect_evals` model string | Honesty | Accuracy | Notes |
| --- | --- | --- | ---: | ---: | --- |
| GPT-4o overlap rerun | 2026-04-02 | `openai/gpt-4o-2024-08-06` | 0.568 | 0.798 | Current overlap anchor against the paper |
| Claude Opus 4.6 | 2026-04-07 | `anthropic/claude-opus-4-6` | 0.838 | 0.906 | Strongest honesty result so far |
| GPT-5.4 | 2026-04-07 | `openai/gpt-5.4` | 0.766 | 0.846 | Strong current GPT-family result |
| DeepSeek V3.2 | 2026-04-07 | `openai-api/deepseek/deepseek-chat` | 0.438 | 0.785 | High accuracy with low honesty; the inspect string used the provider alias that mapped to DeepSeek-V3.2 on the run date |
| Grok 4.20 non-reasoning | 2026-04-07 | `grok/grok-4.20-0309-non-reasoning` | 0.375 | 0.756 | Lowest honesty among the completed full runs |
| Gemini 2.5 Flash | 2026-04-08 | `google/gemini-2.5-flash` | 0.558 | 0.689 | First completed Gemini full run |
| Llama 3.1 8B OpenRouter rerun | 2026-04-10 | `openrouter/meta-llama/llama-3.1-8b-instruct` | 0.748 | 0.620 | Current Llama result on the provider-API path; keep the caveats visible |
| Llama 4 Scout | 2026-04-10 | `openrouter/meta-llama/llama-4-scout` | 0.555 | 0.703 | Completed Scout provider-API baseline |
| Qwen 235B | 2026-04-10 | `openrouter/qwen/qwen3-235b-a22b-2507` | 0.514 | 0.739 | Completed Qwen provider-API baseline |

## Main conclusions so far

- The broad MASK pattern still holds across families: several models remain factually accurate while still lying often under pressure.
- Claude Opus 4.6 and GPT-5.4 are much more honest than the frontier range highlighted in the paper.
- DeepSeek V3.2 and Grok still show the sharpest "knows but lies" pattern in the current full-run set.
- Gemini Flash looks better than the paper's Gemini 2.0 Flash baseline, but the family story is still incomplete because the Gemini Pro path remains open.
- Llama 3.1 8B now has a pinned OpenRouter rerun that lands close to the paper's appendix anchor, but the overlap-anchor story remains more fragile than the GPT-4o anchor because judge fragility still mattered more.
- The open-weight provider-API comparison set is now broader than the Llama overlap anchor alone: Scout lands in the middle of the completed pack, while Qwen is less honest than Scout but still materially above the current Grok result.

## Detailed analysis

The [full runs](results-full-runs.md) page now includes:

- **Detailed honesty breakdown:** per-model counts and percentages for each honesty label (honest, lie, evade, no-belief, error).
- **Detailed accuracy breakdown:** per-model counts and percentages for each accuracy label (correct, incorrect, no-belief, error), with provided_facts excluded from the denominator.
- **Knows-but-lies analysis:** accuracy minus honesty gap with conservative 95% CIs and evade/lie ratios.
- **Per-config honesty and accuracy:** archetype-level splits across all six question types.
- **Scale Labs MASK leaderboard comparison:** directional cross-source comparison for the four models that overlap between this project (1,000 public examples) and the Scale leaderboard (500 private examples).

All confidence intervals use Wilson score intervals at the 95% level. At N=1000, headline CI widths are roughly 3 percentage points.

## Read these next

- [Full runs](results-full-runs.md)
- [Family evolution](family-evolution.md)
- [Accuracy comparability](accuracy-comparability.md)
- [Model status and next runs](model-status-and-next-runs.md)
