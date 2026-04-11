# Cost estimation

Check cost on your own provider tier before you launch a full 1,000-sample run.

## The two-step rule

1. Run a small smoke on your own account.
2. Scale that smoke with the calculator, then compare it against the bundled observed run profiles.

This is safer than copying someone else's bill because provider pricing, discounts, and tier rules change.

## What usually drives cost

MASK cost is usually dominated by three buckets: target-model tokens, binary-judge tokens, and numeric-judge tokens. Judge usage can be a large part of the total, so budget for the whole pipeline. In the completed OpenRouter runs, the OpenAI judges accounted for over 95% of total spend.

## Method 1: Scale your own smoke

This is the best starting point.

1. Run 10 samples.
2. Record target and judge token usage.
3. Scale those numbers to the sample count you want.
4. Add margin for retries.

Example:

```powershell
.\.venv\Scripts\python .\scripts\estimate_mask_cost.py `
  --target-input-tokens 5200 `
  --target-output-tokens 2400 `
  --binary-judge-input-tokens 15800 `
  --binary-judge-output-tokens 3100 `
  --numeric-judge-input-tokens 2300 `
  --numeric-judge-output-tokens 700 `
  --scale-from-samples 10 `
  --scale-to-samples 1000 `
  --target-input-price 1.25 `
  --target-output-price 10.00 `
  --binary-judge-input-price 2.50 `
  --binary-judge-output-price 10.00 `
  --numeric-judge-input-price 1.10 `
  --numeric-judge-output-price 4.40
```

## Method 2: Start from an observed run profile

The package includes observed profiles in [`runtime_profiles.csv`](../../data/runtime_profiles.csv). Use them as scale references, then replace them with your own numbers once you have a smoke run.

`run_profile_label` is a short label for one recorded run in the CSV file. It is a run label for cost estimation, not a provider model id.

Example with a bundled profile:

```powershell
.\.venv\Scripts\python .\scripts\estimate_mask_cost.py `
  --run-profile-label gpt_54_postfix_sync_2026_04_07 `
  --target-input-price 1.25 `
  --target-output-price 10.00 `
  --binary-judge-input-price 2.50 `
  --binary-judge-output-price 10.00 `
  --numeric-judge-input-price 1.10 `
  --numeric-judge-output-price 4.40
```

## Manual token entry example

```powershell
.\.venv\Scripts\python .\scripts\estimate_mask_cost.py `
  --target-input-tokens 500000 `
  --target-output-tokens 200000 `
  --binary-judge-input-tokens 1500000 `
  --binary-judge-output-tokens 300000 `
  --numeric-judge-input-tokens 200000 `
  --numeric-judge-output-tokens 60000 `
  --target-input-price 1.25 `
  --target-output-price 10.00 `
  --binary-judge-input-price 2.50 `
  --binary-judge-output-price 10.00 `
  --numeric-judge-input-price 1.10 `
  --numeric-judge-output-price 4.40
```

All prices are USD per million tokens.

## Google batch note

For Google batch runs, separate the target-generation cost from replay-scoring cost. The bundled Gemini 2.5 Flash full profile and Gemini 3.1 Pro smoke profile include target-generation estimates for the generation stage only.

## A simple budgeting habit

1. Run the smoke.
2. Estimate the cost.
3. Compare it against the balance and limits you actually have.
4. Launch the full run only after that check.
