# Accuracy comparability

This page explains why some results in this package show both a raw accuracy value and a paper-comparable accuracy value.

## Short version

- The current honesty score is broadly comparable to the paper's `1 - P(lie)` score.
- Older raw accuracy values from the March runs are not directly paper-comparable because they were generated with an older `inspect_evals` path that still counted `NO_BELIEF_ELICITATION_DONE` rows in the denominator.
- The clearest paper-comparable overlap anchors are the fixed-main GPT-4o rerun from 2026-04-02 and the pinned Llama 3.1 OpenRouter rerun from 2026-04-10.
- The original March Llama result was not comparable in the same way. The later OpenRouter rerun is.

## Why two accuracy columns exist

The public MASK path contains 274 `provided_facts` samples. Those samples do not go through belief elicitation, because the ground truth is given in the prompt itself. Older raw accuracy calculations kept them in the denominator even though they could not contribute belief-based accuracy in the same way as the rest of the set.

Older local runs such as the March GPT-4o pinned rerun and the original March Llama 3.1 run used `inspect_evals 0.5.1` (released 2026-03-12). The paper-comparable denominator fix landed later through [issue #1380](https://github.com/UKGovernmentBEIS/inspect_evals/issues/1380) and [PR #1385](https://github.com/UKGovernmentBEIS/inspect_evals/pull/1385), merged on 2026-04-07. The current comparable reruns in this package are pinned to commit `0b1a737cf1d4dbd32235f47bf30bd731873dfc27`, which includes that fix.

The practical change is simple: `calculate_accuracy_score` now excludes `NO_BELIEF_ELICITATION_DONE` rows from the denominator by default. That brings the standard `overall_accuracy` headline much closer to the paper's intended interpretation on the public 1,000-example set.

That is why the public tables use:

- `Raw accuracy` for the number produced directly by the run
- `Paper-comparable accuracy` when a denominator-aligned comparison is needed

## Key comparison results

| Model | Run date | Raw accuracy | Paper-comparable accuracy | Paper appendix accuracy | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| GPT-4o pinned rerun | 2026-03-24 | 0.482 | 0.664 | 0.786 | Useful for the diagnosis, but still on the older `inspect_evals 0.5.1` path |
| GPT-4o fixed-main rerun | 2026-04-02 | 0.579 | 0.798 | 0.786 | Best overlap anchor for paper comparison |
| Llama 3.1 original | 2026-03-24 | 0.306 | 0.421 | 0.620 | March self-hosted run on the older `inspect_evals 0.5.1` path; not paper-comparable by default |
| Llama 3.1 OpenRouter rerun | 2026-04-10 | 0.620 | 0.620 | 0.620 | Pinned post-PR-`#1385` provider-API rerun that lands on the paper appendix accuracy anchor |

The main practical conclusion is that the denominator-aligned GPT-4o rerun reaches the same broad region as the paper appendix once `provided_facts` is excluded, and the latest pinned OpenRouter Llama rerun is comparable in the same way.

## What this means for reading the tables

- Use honesty normally, with the usual benchmark caution.
- Use the current post-fix full-run accuracy values as the main operational results.
- Use the GPT-4o fixed-main rerun to understand how comparability was recovered.
- Use the April 10 OpenRouter Llama rerun as the current paper-comparable Llama anchor.
- The public replication still uses the public 1,000-example set rather than the paper's full 1,500-example setup, so it is normal that values are still not exactly the same.
