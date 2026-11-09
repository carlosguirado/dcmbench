# 0.1.x regression baseline

These are the benchmark results produced by dcmbench 0.1.x and its benchmark scripts, recorded before the 1.0 rewrite. Every difference from them after the rewrite must be explained. See `docs/PLAN.md` §7.

## Environment
- Python 3.12, `dcmbench==0.1.2` (PyPI), `biogeme==3.2.14`.
- `tomlkit<0.13.3`. Newer tomlkit makes Biogeme 3.2.14 fail with "Comment cannot contain line breaks".
- Data: the `dcmbench-data` GitHub repo at commit `98c93d2b`, the same commit that `dcmbench.data` pins.

## Files
| File | Contents | Produced by |
|---|---|---|
| `full_sample_estimates.json` | Every estimation in benchmark script 02 (ten models, three datasets, full sample): final LL, parameters, robust s.e., configured vs actually used number of draws | `generate.py`, which runs script 02's `main()` unchanged and records each `estimate()` |
| `holdout_80_20.json` / `.csv` | The same ten models on `train_test_split(test_size=0.2, stratify=CHOICE, random_state=42)`: in-sample and out-of-sample LL, accuracy, macro F1, macro recall | `generate.py` |
| `vot_and_policy.json` | ModeCanada VOT summaries and the Swissmetro car-cost sensitivity numbers | Transcribed from the output of benchmark scripts 03 and 04, run unchanged |

To regenerate, run this from an empty working directory:

```
python generate.py <path to 0.1.x scripts>/02_model_benchmarking.py tests/fixtures/v0_1_baseline
```

## Comparing against this baseline
- **Closed-form models** (MNL, NL, the CUSTOM MNL) should match to the printed precision.
- **MXL models** use simulated likelihoods, so results depend on the draws. In 0.1.x, the draws actually used were 100, not the 200 the scripts set after construction (see `draws_actually_used`). Compare MXL with a tolerance, or with the same draws.
- **Units.** The 0.1.x scripts scaled Swissmetro times and costs by 1/100 in preprocessing. 1.0 keeps data in original units, so 1.0 coefficients differ by that factor. Likelihoods, probabilities and VOTs do not.
