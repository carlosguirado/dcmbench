# DCMBench 1.0 Overhaul Plan

Status: draft, agreed in discussion on 2026-10-20. This is a living document. Update it when decisions change, and record each change in the [decision log](#decision-log).

## 1. Why we are doing this

DCMBench 0.1.2 (PyPI, Aug 2025) describes itself as a model-agnostic benchmarking layer, but the code doesn't yet deliver that. Behavioral outputs depend on Biogeme internals, the JSON specification route barely works, and much of the package is dead or duplicated. The overhaul has three aims:

1. Make the package do what it claims: evaluate any model from its choice probabilities, across engines, in one call.
2. Support model classes that classical estimation packages can't estimate, such as hybrid ML–choice models, and give contributors a clear path to add models and data.
3. Ship it as `dcmbench` **1.0.0** on PyPI.

### Goals and deliverables

| Goal | Deliverable |
|---|---|
| Benchmark models that classical packages can't estimate | Hybrid ML–choice models (TasteNet, ResLogit, RUMBoost) and sklearn baselines alongside MNL/NL/MXL, on identical metrics (§6, Phase 4). |
| Systematic out-of-sample evaluation | Cross-validation and bootstrap rather than a single holdout (validation module, Phase 2). In-sample and out-of-sample results reported side by side. |
| A practical extension path | Models only have to supply probabilities. Each hybrid model's demo script doubles as a worked example. "Add a model" and "Add a dataset" guides (Phase 6). |
| Reach R/Apollo users | A precomputed-predictions engine in the core. An Rscript bridge last and optional (Phase 8). |

### Claims the 1.0 code must back up

| Claim | Status in 0.1.x | Target |
|---|---|---|
| A model only needs to supply choice probabilities | False. Only one adapter implements the interface. VOT string-matches Biogeme beta names. | True for every metric and behavioral output (§5). |
| One JSON spec can be estimated by any supported engine | Biogeme only | Biogeme **and** PyTorch build from the same spec (Phase 3). |
| A single evaluation call returns the full multi-dimensional benchmark | The benchmarker returns fit and accuracy only | `evaluate(...)` returns fit, prediction, VOT, elasticities, scenarios and regularity (Phase 2). |
| VOT by numerical perturbation for models without linear coefficients | Not implemented | Implemented (§5.2). |
| Behavioral regularity (Feng et al. 2024) as the diagnostic for ML models | Not implemented | Implemented (§5.4). |
| Cross-validation and bootstrap | Exist in `dcmbench.validation` (unreleased) | Released, and used throughout the benchmark. |

### Non-goals
- **Estimation.** DCMBench orchestrates estimation and evaluation. It does not compete with Biogeme, Apollo or PyTorch.
- **Choice problems beyond mode choice.** 1.0 focuses on mode choice.
- **Compatibility with the 0.1.x API.** Breaking it is accepted. New benchmark scripts replace the old ones.
- **Generating Apollo code from JSON specs.**

## 2. Design principles

1. **Models only supply probabilities.** A fitted model is anything that maps a DataFrame to an `n_obs × n_alternatives` probability table, in a fixed alternative order. Everything else is computed by DCMBench, often by calling the model on perturbed data.
2. **Each dataset carries a description of its choice problem.** This records the alternatives, choice column, availability, which column holds each alternative's attributes, and units. It is what lets DCMBench perturb "car cost" or compute VOT without knowing anything about the model's internals.
3. **Engine adapters are thin.** An adapter translates between an engine and the core protocol, and nothing more. It contains no dataset-specific logic.
4. **No silent fallbacks.** When something fails, it raises with a clear message. A metric that doesn't apply is reported as "not applicable" (for example, ρ̄² for a random forest), never as 0, None or an equal-probability guess.
5. **The validation module is the core, kept intact.** hphoebe's `FittedModel` protocol is already "probabilities in, metrics out". The new core conforms to it, not the other way round (§3.6).
6. **Correctness is tested against known truth.** We use synthetic data with known parameters, closed-form results, and cross-engine parity.

## 3. Target architecture

### 3.1 Package layout

```
dcmbench/
  problem.py          # ChoiceProblem: alternatives, attributes, availability, units
  protocols.py        # re-exports validation protocols + optional capability protocols
  data/               # dataset registry, loader, cache, checksums, per-dataset ChoiceProblem
  specs/              # JSON spec schema + validation
    biogeme.py        #   spec -> Biogeme model (MNL, NL, MXL)
    torch.py          #   spec -> PyTorch model (MNL, NL, MXL)
  engines/            # thin adapters to the core protocol
    biogeme.py        #   Biogeme fitter / fitted model
    torch.py          #   generic nn.Module adapter + training helper
    sklearn.py        #   any classifier with predict_proba
    precomputed.py    #   probabilities supplied as a table (Apollo, any other engine)
    apollo.py         #   Rscript bridge (Phase 8, optional)
  evaluation/
    run.py            # evaluate(): one call -> full multi-dimensional result
    predict.py        # the checked model boundary; determinism check
    perturb.py        # attribute perturbation engine (shared by everything below)
    elasticity.py     # point elasticities, own/cross, individual/aggregate
    vot.py            # VOT distributions (§5.2)
    scenarios.py      # policy scenarios, predicted shares, diversion, arc elasticities
    regularity.py     # behavioral regularity (§5.4)
  validation/         # hphoebe's module, kept as is (CV, bootstrap, OOS metrics)
  reporting/
    card.py           # specification card (machine-readable) + rendered view
    tables.py, plots.py
examples/             # demo + reproducibility scripts (TasteNet, ResLogit, RUMBoost live here)
docs/                 # user guide, "Add a model", "Add a dataset", API reference
tests/
```

Note that `.gitignore` currently ignores `examples/` and `*.png`. Both entries have to be changed before we commit demo scripts or doc figures.

### 3.2 Core interfaces

These are specified in `docs/design/phase1-core.md`:
- `ChoiceProblem` and `Attribute`;
- the model protocols: the validation module's `FittedModel` and `ModelFitter`, plus the optional `ProvidesCoefficients` and `ProvidesMarginalUtilities`;
- result objects.

`evaluate()` takes fitters or fitted models, a dataset and its `ChoiceProblem`, a protocol (holdout, CV or bootstrap) and a set of behavioral analyses. It returns one result object, and that object can be written out as a specification card.

### 3.4 Model specification format (spec v2)

**What's wrong with v1.** The JSON route never worked beyond the basics:
- **Only one spec was ever published** (`mnl_swissmetro.json` in `dcmbench-data`). The remote `metadata.json` also lists `nested_swissmetro.json` and `mixed_swissmetro.json`, which don't exist. None of the 0.1.x benchmark models were built from JSON.
- **MXL is broken.** `random_parameters` is read and ignored. No draws enter the utilities, so the "MXL" is an MNL wrapped in `MonteCarlo`.
- **Formulas run through Python `eval()` as Biogeme code.** This has three consequences:
  - The same spec can't build a PyTorch model.
  - Functions such as `log()` aren't available, so the CUSTOM spec's nonlinear cost terms can't be expressed.
  - It's a security hole for specs fetched from URLs. Availability strings go through an *unrestricted* `eval()`.
- **MNL and NL do work.** They are exercised by `tests/test_validation.py`.

**v2 design:**
- **An expression language that doesn't depend on any engine.** Formula strings are parsed with Python's `ast` module into an allow-listed tree:
  - numbers, parameter names and data columns;
  - `+ − * / **` and unary minus;
  - `log`, `exp`, and possibly `boxcox`;
  - comparisons used as 0/1 indicators (e.g. `(INCOME > 3)`), for segments and interactions.

  Anything else is rejected with a clear error. The same tree compiles to a Biogeme expression or a PyTorch function. That compilation step is what makes "one spec, two engines" true.
- **Model types:**
  - MNL.
  - NL, with one or more nests.
  - MXL, with random coefficients (normal, lognormal, and probably uniform and triangular), a configurable number of draws and seed, and optional panel structure (one person, many choices).
  - Latent class and cross-nested logit are not in v2. They stay in hand-written scripts.
- **Linked to the dataset's choice-problem description.** Alternatives are referred to by name (`"train"`, `"car"`), matching the dataset's `ChoiceProblem`, instead of codes `"1"`, `"2"`, `"3"`. Availability refers to columns or constants only.
- **Versioned and validated.** A `spec_version: 2` field, and a schema check with clear errors on load.
- **Shipped in the repo.** The specs behind the benchmark models live in the repo (package data or `examples/specs/`), as well as in `dcmbench-data`.
- **Tests:**
  - Parser tests, including rejection of disallowed constructs.
  - Biogeme-vs-PyTorch parity for MNL, NL and MXL (§7).
  - A known-truth test: an MXL estimated on synthetic MXL data gets its σ back.

Rough effort: 1–1.5 weeks, an estimate. The hard part is MXL parity between engines, whose random draws differ.

### 3.5 Specification card (prototype)

A "specification card" is a structured, machine-readable summary of what a model specification implies: fit, prediction, economic valuations, policy response and regularity. DCMBench produces a **prototype** card. It is a versioned strawman meant to seed community discussion of reporting conventions, not a claim to be a standard.

- **Provenance:**
  - dataset name, version and checksum;
  - spec (or model code) hash;
  - engine and version, DCMBench version;
  - seeds and validation protocol (holdout, CV or bootstrap, with settings).
- **Statistical fit:** in-sample LL, ρ², ρ̄² (or "not applicable"), number of parameters, convergence.
- **Prediction:** out-of-sample LL, macro F1, recall and accuracy, with spread across CV folds or bootstrap draws.
- **Economic inference:**
  - which VOT route was used (§5.2);
  - quantiles;
  - share of negative values;
  - share above a user-supplied plausibility bound.
- **Policy:** shares and diversion ratios under a standard perturbation, which is configurable and recorded on the card.
- **Behavioral regularity:** Feng et al.'s strong and weak metrics (§5.4).
- **Format:** a machine-readable JSON file validated by a versioned schema (`card_version`), plus a rendered Markdown/HTML view. Cards from different runs can be loaded and compared side by side.

### 3.6 What stays in `validation/`

The rule: public API unchanged, and the existing tests (`tests/test_validation.py`) stay unmodified and must pass at every phase. They are a CI gate.

The unavoidable touches:
- **`BiogemeSpecFitter` internals.** These sit on `UniversalBiogemeAdapter`, which is being removed. They will be rebuilt on `engines/biogeme.py` with the same behavior. (The Biogeme 3.3 constructor-parameter fix already made in `estimators.py` is part of this.)
- **`n_params: int | None`.** RUMBoost, neural networks and sklearn models have no meaningful parameter count. ρ̄² then reports "not applicable". This is a small, backwards-compatible protocol change. Flag it to hphoebe as a courtesy.
- **Shared metrics.** `validation/metrics.py` becomes the single home of the likelihood and classification metrics. The evaluation layer imports from it rather than duplicating it.

## 4. What we keep and what we delete

| 0.1.x module | Fate |
|---|---|
| `validation/*` | **Keep** (see §3.6) |
| `datasets/dataset_loader.py` | Salvage the download/cache logic into `data/`. Fix the repo-name mismatch (`dcmbench-data` vs `dcmbench-datasets`). Make caching the default. Remove `logging.basicConfig` at import. |
| `model_specifications.py` | Replace with `specs/` (spec v2, §3.4). Delete the `eval()` parsing, the broken MXL branch and the dead helpers. Remote fetching is kept, with checksums. |
| `utils/metrics.py` | Salvage the finite-difference elasticity helpers into `evaluation/behavioral/` |
| `utils/sensitivity_analysis.py` (`SensitivityAnalysis` only) | Salvage the perturbation idea into `perturb.py` |
| `adapters/pytorch_adapter.py`, `adapters/sklearn_adapter.py` | Rewrite as thin `engines/` adapters |
| `adapters/universal_biogeme_adapter*.py`, `biogeme_adapter.py`, `enhanced_biogeme_adapter.py`, `biogeme_vot_adapter.py`, `utils/biogeme_wrapper.py`, `utils/model_extensions.py` | **Delete.** These are five overlapping Biogeme paths, with dataset-specific fallbacks. |
| `adapters/factory.py`, `pytorch_model.py`, `sklearn_model.py`, `datasets/simple_fetcher.py` | **Delete** (broken imports or never imported) |
| `utils/vot_*.py`, `individual_parameters.py`, `policy_analysis.py`, rest of `sensitivity_analysis.py` | **Delete.** They are replaced by `evaluation/behavioral/`. Port any plotting worth keeping into `reporting/plots.py`. |
| `model_benchmarker/*` | **Delete.** Replaced by `evaluate()` and `reporting/`. |
| `setup.py`, `requirements.txt`, `GITHUB_SETUP.md`, `PUBLISHING_GUIDE.md` | **Delete.** `pyproject.toml` is the single source of packaging config. |

## 5. Behavioral outputs from probabilities alone

All of these run on the shared perturbation engine. Given a `ChoiceProblem`, it modifies an attribute of one or more alternatives (for example "car cost × 1.1") and re-predicts. Availability is respected throughout.

### 5.1 Elasticities and policy scenarios
- **Point elasticities.** Central finite differences, with a relative step. The step size is configurable and reported.
- **Arc elasticities.** Used for discrete policy changes (for example a 20% toll increase), and always for piecewise-constant models (RUMBoost, tree-based sklearn models).
- **Levels.** Elasticities are computed per person and aggregated by probability weighting. Both own and cross elasticities are reported.
- **Policy scenarios.** Predicted shares before and after the change, plus diversion ratios (where displaced demand goes). This generalizes the 0.1.x car-cost sensitivity analysis to any model.

### 5.2 VOT distributions: two routes

A VOT distribution is a set of individual VOT values with weights. Each model supplies it in one of two ways.

**Route A: from the model's parameters.** Used when the model implements `ProvidesMarginalUtilities`: VOT is the ratio of marginal utilities of time and cost, per observation and draw.

| Model | What it supplies | Shape |
|---|---|---|
| MNL | β_time / β_cost | a single spike |
| Segmented MNL | one value per segment, weighted by segment share | a few spikes |
| Latent class | each class's value, weighted by the person's class-membership probabilities | K weighted spikes |
| MXL | ratios from draws of the estimated taste distribution (population view). Optionally, each person's posterior expected tastes given their choices (Train 2009, ch. 11). | continuous, with tails |
| TasteNet | each person's coefficients as output by the network | empirical distribution over people |

**Route B: from probability derivatives.** The fallback for everything else (ResLogit, RUMBoost, sklearn):
VOT_ij = (∂P_ij/∂time_ij) / (∂P_ij/∂cost_ij), computed per person and alternative.
- For logit models whose utility is linear in time and cost, this equals β_time/β_cost exactly. That makes it a built-in test.
- For MXL, Route B collapses the distribution to roughly one value per person. It is therefore **not** used for MXL.
- For piecewise-constant models, derivatives are zero or undefined. Use arc differences with a reported step, and flag the result.

Every VOT result records which route produced it. The 0.1.x ModeCanada VOT density comparison is generalized to a single comparison across models: one quantity, VOT, under different theories of how preferences vary, from classical to hybrid models. Which models even have a well-defined VOT distribution becomes a reported result in its own right.

### 5.3 Units
VOT is reported in `cost unit / hour`, using the `ChoiceProblem` units. This replaces the old hard-coded "× 60, CHF/h".

### 5.4 Behavioral regularity (Feng et al. 2024)
Behavioral regularity is the natural diagnostic for ML and hybrid models. With those models in the benchmark, it becomes a reported result.

**Definition** (Feng et al. 2024, Eqs. 5–6; [arXiv:2404.14701](https://arxiv.org/abs/2404.14701); code: github.com/siqi-feng/DNN-behavioral-regularity). For alternative *i* and its own cost *d*, regularity is the share of observations where the individual choice probability P_ni falls when x_nd rises:
- **Strong:** ΔP < −τ, which requires a strict decrease.
- **Weak:** ΔP < +τ, which also lets a zero response pass.
- The paper only says "a small negative / positive number". The details below come from their code.

**Their computation**, reproduced exactly:
- One-sided forward difference at each observed point: ΔP = P_i(x_d + h) − P_i(x_d).
- Step h = (max − min of x_d over the full dataset) / 200, in raw units. Any features derived from x_d are recomputed.
- The threshold is applied to the raw ΔP (not ΔP/Δx), with τ = range_d / K. They used K = 1e6 on their Chicago data, and 5e5 or 2e6 on some London subsamples.
- Headline numbers are on test data, and are averaged over 10 training replications.
- They check one pair per dataset: P(car) w.r.t. car cost (once, P(transit) w.r.t. transit cost). Time, other alternatives and cross effects are not checked.
- Reference values for sanity checks: MNL strong 0.980–0.998, weak 1.000. DNN strong 0.185–0.950, depending on sample size and whether the test data is out of domain.

**DCMBench implementation:**
- **Paper-comparable headline:** P(car) w.r.t. car cost, with h = range/200 and τ = range/1e6, on held-out data. Averaged over replications for seeded models.
- **Also reported:** every (alternative, own-cost) pair separately, plus own-time pairs as an extension. Each is labeled.
- **Stored with every result:** the raw ΔP values, h and τ.

**Deviations, each flagged on the card:**
- more than one pair, and any average across pairs;
- one τ convention (K = 1e6) on every dataset;
- time attributes;
- excluding observations where alternative *i* is unavailable. The paper never deals with availability: an unavailable alternative has ΔP = 0, which would count as a strong failure.

**Pitfalls:**
- **Piecewise-constant models** (trees, RUMBoost). With h = range/200 most ΔP are exactly 0. Strong then measures how often a split falls inside [x, x+h], and weak is close to 1. Report the plain forward difference as specified, with a flag. Any larger-step variant is a separate, labeled extension.
- **Simulated probabilities** (MXL). Both evaluations must use the same draws (common random numbers). Otherwise, simulation noise swamps a τ of about 1e-5. This becomes a requirement on engines: `predict_probabilities` must be deterministic for a fixed fitted model.
- **Costs zeroed by the spec** (e.g. Swissmetro GA holders). Perturbing raw cost correctly gives ΔP = 0 for those observations. They pass weak and fail strong. Report how many observations are structurally unaffected.

**Test cases:** a correctly signed MNL scores weak = 1 and strong ≈ 1, and a deliberately non-monotone synthetic model does not.

- **Out of scope:** Feng et al.'s remedy, gradient regularization, is an estimation technique, not an evaluation.

## 6. Model coverage

| Model | Engine | Where the model code lives | Notes |
|---|---|---|---|
| MNL, NL, MXL | Biogeme | `specs/biogeme.py` (from JSON) | Classical baselines |
| MNL, NL, MXL | PyTorch | `specs/torch.py` (from the same JSON) | Parity with Biogeme required (§7) |
| CUSTOM Swissmetro | Biogeme (+ PyTorch if the spec allows) | JSON spec | As in the 0.1.x benchmark scripts, with corrections |
| TasteNet | PyTorch | `examples/` | Reimplemented from Han et al. (2022) unless official code turns up |
| ResLogit | PyTorch | `examples/` | Reimplemented from Wong & Farooq (2021) unless official code turns up |
| RUMBoost | `rumboost` (PyPI) | `examples/` + thin wrapper | Check whether its paper reports LPMC results to compare against |
| sklearn baselines (e.g. random forest, MLP, gradient boosting) | sklearn | `engines/sklearn.py` | Availability masking + renormalization required; ρ̄² not applicable |
| Latent class | Biogeme or PyTorch | `examples/` | **Optional.** Mixture likelihood written by hand. Cut if time is short. |
| Apollo models | `precomputed` / `apollo` | user's R script | Phase 8 |

Fidelity of the reimplementations matters. Each one gets a short note comparing our results with those in its original paper, wherever comparable data exists.

## 7. Testing strategy

- **Validation gate.** `tests/test_validation.py` passes unchanged at every phase.
- **Known truth.** Synthetic MNL, NL and MXL data generated from known parameters must give back:
  - the parameters (within sampling tolerance);
  - the exact VOT through both routes, for MNL;
  - the closed-form negative-VOT share Φ(−|μ|/σ), for normal MXL;
  - closed-form MNL elasticities.
- **Cross-engine parity.** Biogeme and PyTorch MNL/NL/MXL built from the same JSON give the same LL and estimates on the same data, within tolerance.
- **Regression baseline.** Before any rewrite, reproduce the 0.1.x benchmark results with the 0.1.x code. Results become fixtures (`tests/fixtures/v0_1_baseline/`). Any difference after the rewrite must be explained (for example, by the Biogeme 3.3 changes).
- **No-fallback tests.** A failing model raises an error. It never returns degraded output.
- **CI.** GitHub Actions on Python 3.12 and 3.13. The core and engine tests run on every push. Slow tests (MXL, hybrids) are marked and run nightly or on demand. An R job is added only if Phase 8 happens.

## 8. Packaging and release

- `pyproject.toml` only. `requires-python >= 3.12`.
- Dependencies:
  - Core: numpy, pandas, scipy, scikit-learn, matplotlib, requests.
  - Extras:
    - `[biogeme]` (biogeme ≥ 3.3)
    - `[torch]`
    - `[rumboost]`
    - `[all]`
    - `[dev]`
  - Biogeme is an optional extra (decided in Phase 1).
- Declare every import. `tabulate` and `tqdm` are imported but undeclared today; drop them if no longer used.
- No files written to the working directory by default. This means turning off Biogeme's `biogeme.toml`, HTML and YAML output during orchestrated runs.
- Docs on Read the Docs or GitHub Pages: user guide, "Add a model", "Add a dataset", the specification card, API reference.
- Release 1.0.0 to TestPyPI, install it in a clean environment, run the examples, then publish to PyPI. Add a CHANGELOG entry with a migration note from 0.1.x.

## 9. Phases

Each phase is a pull request into a long-running `v1` branch. `main` stays at 0.1.x until release.

| # | Phase | Deliverables | Done when |
|---|---|---|---|
| 0 | **Baseline** | Tag current `main`. Move the uncommitted Biogeme 3.3 fixes onto `v1`. Reproduce the 0.1.x benchmark results. Record fixtures. | Results reproduced, or every discrepancy explained |
| 1 | **Core design** | `ChoiceProblem`, protocols, result objects. Data layer with checksums and a `ChoiceProblem` for Swissmetro, ModeCanada and LPMC. A short design note in `docs/` for sign-off. | Design approved; data layer tested |
| 2 | **Evaluation engine** | `evaluate()`, perturbation engine, elasticities, scenarios, VOT (both routes), regularity, all wired to `validation/` | Known-truth tests pass |
| 3 | **Engines + specs** | Spec v2: parser, schema, Biogeme and PyTorch compilers for MNL/NL/MXL (§3.4). The benchmark specs written as v2 JSON. sklearn and precomputed engines. `BiogemeSpecFitter` rebuilt. Old adapters deleted. | Parser and parity tests pass; validation gate passes; every benchmark model that v2 can express builds from JSON |
| 4 | **New models** | RUMBoost wrapper. TasteNet and ResLogit in `examples/`, with fidelity notes. Latent class if time allows. | Each runs through `evaluate()` on all three datasets |
| 5 | **Reporting** | Prototype specification card (§3.5): JSON schema, writer, loader, rendered view, side-by-side comparison. Tables and plots, including the generalized VOT comparison and policy sensitivity plots. | A schema-valid card is produced for every benchmark model |
| 6 | **Docs + release** | Guides, API docs, packaging, CI, 1.0.0 on TestPyPI, then PyPI | Clean-environment install runs all examples |
| 7 | **Benchmark reproducibility** | Scripts regenerating every benchmark table and figure, with CV instead of a single holdout, and hybrid and ML models added | All benchmark numbers come from scripts in the repo |
| 8 | **Apollo bridge** (optional) | Rscript bridge + R template. Apollo–Biogeme MNL parity test. | Parity holds; CV runs through Apollo |

Progress:
- **Phase 0: done 2026-10-20.** Fixtures are in `tests/fixtures/v0_1_baseline/`.
- **Phase 1: done 2026-10-20.**
  - What was built: `dcmbench.problem`, `dcmbench.data` (Swissmetro, ModeCanada, LPMC), `dcmbench.protocols`, `dcmbench.results`, and `n_params: int | None` in `validation/`.
  - Tests: offline tests, plus opt-in network tests (`DCMBENCH_NETWORK_TESTS=1`) that reproduce the 0.1.x benchmark samples, including the exact LPMC rows.
  - Design: `docs/design/phase1-core.md`.
- **Phase 2: done 2026-10-20.**
  - What was built: `dcmbench.evaluation`, covering `evaluate()` with holdout, CV and bootstrap; elasticities; VOT by both routes; scenarios and sensitivity curves; and behavioral regularity. Macro F1 and recall were added to the validation metrics.
  - Tests: known-truth tests (MNL closed forms, normal-MXL negative-VOT share, IIA diversion, regularity counts).
  - Real-data check: a throwaway MNL reproduces the 0.1.x Swissmetro baseline exactly.
  - Design: `docs/design/phase2-evaluation.md`.

Ordering note: benchmark scripts can be drafted against the `v1` branch from Phase 4 on. They need 1.0.0 to be published only if results must cite a release.

### 9.1 Phase 0 inputs: what the 0.1.x benchmark scripts tell us

These come from a review of the 0.1.x benchmark scripts (00–04, not in this repo).

**What the scripts actually used.**
- All ten models are hand-written Biogeme code. `fetch_model_spec` and `build_model_from_spec` are imported in script 02 but never called.
- dcmbench was used only for:
  - data loading;
  - `UniversalBiogemeAdapter` and `SimpleBenchmarker`, for metrics;
  - `PolicyAnalyzer`, for ASC calibration and sensitivity analysis;
  - the VOT calculator and plotter.
- The PyTorch MNL (script 01) is written by hand in the script.

**Settings to carry over into spec v2 and the datasets' choice-problem descriptions:**
- **Swissmetro:**
  - Filter `PURPOSE ∈ {1,3}` and `CHOICE > 0` (N = 6768).
  - Time and cost ÷ 100. Train and Swissmetro cost set to 0 when the respondent has a GA (annual travel pass).
  - Swissmetro is the reference alternative.
- **ModeCanada:**
  - Long-to-wide conversion. Time = in-vehicle + out-of-vehicle time.
  - An alternative is available if its row exists. N = 4324. Car is the reference.
- **LPMC:**
  - All four alternatives treated as available.
  - `sample(n=5000, random_state=42)` on the full table.
- **MXL:** pseudo-random draws, no panel. The scripts set 200 draws, but 100 were used (see the fixtures).
- **Hidden preprocessing.** `fetch_data` silently runs `dropna()` and re-downloads unpinned data from GitHub `main` on every call. The new data layer makes preprocessing explicit and pins data versions.

**Phase 0 result (2026-10-20).** Scripts 00, 02, 03 and 04 were run unchanged in a 0.1.x environment, and the same models were also estimated on an 80/20 stratified holdout. The results are the regression baseline in `tests/fixtures/v0_1_baseline/`. Several model specifications in those scripts need correcting in the 1.0 specs (Phase 3): nest-parameter bounds, an interaction that is always zero, the income coding, and the number of draws. MXL panel structure is to be decided.

## 10. Risks and open questions

- **Reimplementation fidelity** (TasteNet, ResLogit). Before writing either, do a proper search for official code. Document architecture choices and compare with the original papers.
- **VOT methodology.** Route A versus Route B (§5.2) is a methodological choice. It must be documented, and every result must say which route it used.
- **MXL in PyTorch.** The simulated likelihood must match Biogeme's draws closely enough for parity. Use a tolerance and a fixed number of draws.
- **Availability in ML models.** sklearn and NN models must mask unavailable alternatives, or the LL and F1 results are invalid.
- **Data hosting.** Datasets live in a separate GitHub repo. Decide whether to keep that, move to Zenodo (which gives a DOI), or both.
- **Runtime.** MXL, hybrid models, CV and bootstrap together can be slow. Fitting folds in parallel and caching results may be needed.
- **Feng et al. definitions.** Resolved 2026-10-20 (§5.4). Their metric is a finite difference on probabilities, so it can be computed for any model; the deviations are listed and flagged.
- **Specification card overclaiming.** Frame the card as a prototype for discussion (§3.5). Standardizing metrics needs community consensus.
- **Repo hygiene.** Private working materials (`paper/`) are in `.gitignore` and must stay out of the repo.

## Decision log

| Date | Decision |
|---|---|
| 2026-10-20 | Quality over speed; there is no deadline pressure. |
| 2026-10-20 | Release as `dcmbench` 1.0.0 on PyPI. Breaking the 0.1.x API is accepted. New benchmark scripts will be written. |
| 2026-10-20 | Python ≥ 3.12, Biogeme ≥ 3.3. |
| 2026-10-20 | Scope: mode choice only. |
| 2026-10-20 | Hybrid models: TasteNet, ResLogit, RUMBoost. Their code lives in demo scripts, not the library. L-MNL is out. |
| 2026-10-20 | sklearn ML baselines are in. Latent class is optional. |
| 2026-10-20 | JSON specs build both Biogeme and PyTorch models. The PyTorch engine is DCMBench's own, not torch-choice. |
| 2026-10-20 | VOT: Route A where a model has parameters, Route B (flagged) otherwise. A generalized VOT comparison across models. |
| 2026-10-20 | `validation/` stays intact; its tests gate every phase. |
| 2026-10-20 | Apollo: precomputed-predictions engine in the core; Rscript bridge last and optional. |
| 2026-10-20 | Behavioral regularity: implement Feng et al.'s strong and weak metrics exactly (forward difference, h = range/200, τ = range/1e6 on ΔP, own cost, held-out data); extensions and deviations are flagged; gradient regularization is out of scope. |
| 2026-10-20 | DCMBench produces a versioned *prototype* specification card, framed as a strawman for community consensus. |
| 2026-10-20 | Rebuild JSON specs as spec v2: an engine-neutral expression language compiled to Biogeme and PyTorch; MXL done properly; no `eval()`. |
| 2026-10-20 | Private working materials kept out of the repo (`.gitignore`). |
| 2026-10-20 | Phase 1 design accepted (`docs/design/phase1-core.md`): data kept in original units, with scaling in specs; central availability masking for engines that can't handle it; GitHub data pinned by commit (Zenodo decided at Phase 6); Biogeme becomes an optional extra; VOT Route A defined via marginal utilities; datasets named `swissmetro`, `modecanada` and `lpmc` (alias `ltds`). |
| 2026-10-20 | The Chicago mode-choice dataset is not added to the analysis. |
