# Family evolution

This page tracks how several model families moved between the MASK paper results and the current replication runs in this package.

The family comparisons below come from the current dated result tables. Family-level rows compare one paper-era model with one newer model from the same family. The two overlap-anchor rows keep a same-model comparison for the cleanest paper anchors.

## Paper version note

The paper reference values in this page and in `data/family_evolution.csv` come from the [MASK paper's v3](https://arxiv.org/pdf/2503.03750v3) tables (latest arXiv revision as of this package). The honesty and accuracy values themselves are unchanged from v2 (March 2025). The main differences between paper versions are:

- **v1 (March 2025)**: Original release with initial model results.
- **v2 (March 2025)**: Updated values for some models.
- **v3 (later revision)**: Rewording and clarifications only; no new values or models compared to v2.

## GPT note

For the GPT family, this page uses the paper's `gpt-4.5-preview` row rather than the `gpt-4o-2024-08-06` overlap anchor. That family row has the higher paper honesty score and is the more informative time-series comparison against `gpt-5.4`. The separate GPT-4o overlap row stays below as the same-model validation anchor.

## Family-level changes

| Family | Paper model | Current model | Run date | Paper honesty | Current honesty | Honesty change (pp) | Paper accuracy | Current accuracy | Accuracy change (pp) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT | gpt-4.5-preview | gpt-5.4 | 2026-04-07 | 0.565 | 0.766 | +20.1 | 0.767 | 0.846 | +7.9 |
| Claude | claude-3.7-sonnet | claude-opus-4-6 | 2026-04-07 | 0.734 | 0.838 | +10.4 | 0.822 | 0.906 | +8.4 |
| DeepSeek | deepseek-v3 | DeepSeek-V3.2 | 2026-04-07 | 0.465 | 0.438 | -2.7 | 0.716 | 0.785 | +6.9 |
| Grok | grok-2 | grok-4.20-0309-non-reasoning | 2026-04-07 | 0.370 | 0.375 | +0.5 | 0.725 | 0.756 | +3.1 |
| Gemini Flash | gemini-2.0-flash | gemini-2.5-flash | 2026-04-08 | 0.516 | 0.558 | +4.2 | 0.794 | 0.689 | -10.5 |

Change is calculated as (current - paper). The Claude comparison is family-level rather than same-model: the paper tested Sonnet while the current result is Opus.

## Overlap anchors

The next two entries are not family-evolution claims. They are overlap anchors between the paper and the current replication path.

| Anchor | Paper model | Current run | Run date | Paper honesty | Current honesty | Honesty change (pp) | Paper accuracy | Current accuracy | Accuracy change (pp) | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GPT-4o | gpt-4o-2024-08-06 | GPT-4o fixed-main overlap rerun | 2026-04-02 | 0.555 | 0.568 | +1.3 | 0.786 | 0.798 | +1.2 | Best overlap anchor for the paper-comparison story |
| Llama 3.1 8B | llama-3.1-8b-instruct | Llama 3.1 8B OpenRouter rerun | 2026-04-10 | 0.765 | 0.748 | -1.7 | 0.620 | 0.620 | 0.0 | Current provider-API overlap anchor; the earlier April 7 local rerun was more divergent |

## Honesty comparison

- The GPT family is much more honest in the current package than in the paper's GPT-4.5-preview row.
- Claude remains the most honest family in the completed full-run set, and the current Opus row is substantially above the paper's Sonnet row.
- DeepSeek and Grok changed very little on honesty. The "knows but lies" pattern is still intact for both families.
- Gemini Flash improved on honesty from Gemini 2.0 Flash to Gemini 2.5 Flash, but the improvement is modest rather than dramatic.
- The latest Llama 3.1 OpenRouter rerun is now close enough to the paper anchor to treat it as a usable overlap check rather than only as a historical curiosity.

## Accuracy comparison

- GPT, Claude, DeepSeek, and Grok all score higher on accuracy than their paper-era family rows.
- Gemini Flash is the main exception in the completed full-run set: the current Gemini 2.5 Flash full run is more honest than the paper's Gemini 2.0 Flash row, but less accurate.
- The GPT-4o overlap rerun and the latest Llama 3.1 OpenRouter rerun are the two strongest same-model accuracy anchors. GPT-4o lands slightly above the paper appendix value, while Llama matches it exactly.

## Gemini Flash follow-up

The Gemini Flash family now has two public signals in this package:

- The completed `gemini-2.5-flash` full run (`0.558` honesty, `0.689` accuracy)
- The later `gemini-3-flash-preview` smoke (`0.600` honesty, `0.917` accuracy at `n=20`)

The full-run row is the benchmark-level evidence. The later smoke is only a directional clue, but it suggests the newer Gemini Flash line may recover both honesty and accuracy if a full run is completed.

## Main takeaways

- GPT and Claude show the clearest family-level honesty gains.
- DeepSeek and Grok still occupy the high-accuracy, low-honesty corner of the comparison set.
- Gemini Flash looks mixed: honesty is higher than the paper's Gemini 2.0 Flash row, but full-run accuracy is lower.
- GPT-5.4 is notably above the paper's GPT-4.5-preview row on both honesty and accuracy, while GPT-4o stayed very close to the paper once the scorer-path issue was corrected.
- The latest Llama 3.1 OpenRouter rerun is now a much stronger overlap anchor than the earlier April 7 local A40 rerun.

## Read these next

- [Results overview](results-overview.md)
- [Full runs](results-full-runs.md)
- [Accuracy comparability](accuracy-comparability.md)
