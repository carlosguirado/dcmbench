# Phase 1 design: choice problems, datasets, protocols and results

Status: **accepted 2026-10-20**. All seven decisions in §7 were accepted as recommended; Chicago is not added. Covers the deliverables of Phase 1 in `docs/PLAN.md` §9. Section numbers like "Plan §5.2" refer to that document.

## 0. What this note decides

1. How a dataset describes its choice problem (`ChoiceProblem`), and what form the data takes.
2. How datasets are versioned, downloaded, cached and prepared (`dcmbench.data`).
3. The model protocols: hphoebe's `FittedModel`, plus optional capabilities.
4. The shape of evaluation results.
5. What gets built in Phase 1, and what waits.

Decisions that need your sign-off are marked **[decide]** and collected in §7.

## 1. Facts about the datasets that drive the design

From the files in `dcmbench-data` (`main`, as of 2026-10-20):

| | Swissmetro | ModeCanada | LPMC (`ltds`) | Chicago (`chicago_mode`) |
|---|---|---|---|---|
| Rows | 10,728 | 15,520 (**long** format) | 81,086 | 26,100 (one empty row) |
| Observations | 10,728 | 4,324 cases | 81,086 trips | 26,099 |
| Format | wide | long (`case`, `alt`) | wide | wide |
| Alternatives | train, SM, car (`CHOICE` 1/2/3; 0 = unknown, 9 rows) | train, car, bus, air (row exists = available) | walk, cycle, pt, drive (`travel_mode` strings) | `mode` 0/1/2 (labels undocumented) |
| Availability | `TRAIN_AV`, `SM_AV`, `CAR_AV` (car unavailable in 1,683 rows) | presence of the row (`noalt` = 2–4) | none given (all four treated as available) | none given |
| Time units | minutes | minutes (`ivt`, `ovt`) | **hours** (`dur_*`) | minutes (by the look of the values) |
| Cost units | CHF (0 for a GA holder, if the model says so) | CAD | GBP | USD (presumably) |
| Respondent ID (panel) | `ID` (1,192 people × 9 choices) | none (each case is its own) | `household_id` + `person_n` | none |
| Coding traps | INCOME 4 = unknown; PURPOSE codes | `urban` has three values (0/1/2) | categorical strings | index column named `chicago_mode.csv` |

Three things follow from this:
- **Data must stay in its original units.** Units differ across datasets, and LPMC is in hours. dcmbench 0.1.x assumed minutes for LPMC's VOT.
- **Long-format data must be converted.** ModeCanada has to be made wide in one documented, tested place.
- **Panel identifiers must be declared.** Both CV grouping (the validation module already supports `group_column`) and a panel MXL depend on them.

Chicago is in the data repo but unused, and its provenance is unknown. **[decide]** Should it be added as a fourth mode-choice dataset? It would broaden the mode-choice benchmark, but it needs a citation, a license and a codebook.

## 2. `ChoiceProblem`

A frozen description of one choice problem over one wide-format table. It never modifies the data.

```python
@dataclass(frozen=True)
class Attribute:
    columns: Mapping[str, str]       # alternative -> column, e.g. {"train": "TRAIN_TT", "car": "CAR_TT"}
    unit: str                        # "minute", "hour", "CHF", "GBP", ... (see §2.2)
    kind: Literal["time", "cost", "other"] = "other"

@dataclass(frozen=True)
class ChoiceProblem:
    name: str
    alternatives: tuple[str, ...]            # fixed order = probability column order
    choice_column: str
    choice_codes: Mapping[Any, str]          # raw value in choice_column -> alternative
    availability: Mapping[str, str | None]   # alternative -> 0/1 column; None = always available
    attributes: Mapping[str, Attribute]      # "time", "cost", "headway", ...
    group_column: str | None = None          # respondent id for panel data / grouped CV
    weight_column: str | None = None

    def validate(self, data: pd.DataFrame) -> None: ...      # columns exist, codes valid, avail 0/1, chosen alt available
    def choices(self, data) -> pd.Series: ...                # alternative names, aligned with data
    def availability_matrix(self, data) -> pd.DataFrame: ... # n_obs × alternatives, bool
```

### 2.1 Design choices
- **Alternatives are named; raw codes stay untouched.** `choice_codes` maps raw values to names (`{1: "train", 2: "sm", 3: "car"}`). Probability tables from every model use `alternatives` as their columns. This mapping is exactly what hphoebe's `InternalValidator(choice_mapping=...)` already takes, so the validation module can be driven straight from a `ChoiceProblem` without changes.
- **The data stays in original units. Scaling moves into the specs.** The 0.1.x benchmark scripts divided Swissmetro times and costs by 100, and zeroed train/SM costs for GA holders, in preprocessing. In 1.0, the spec expresses this, e.g. `B_COST * TRAIN_CO * (GA == 0) / 100`. Spec v2 (Plan §3.4) can express this. Two consequences:
  - The perturbation engine perturbs real, unit-labeled columns. A "+10% car cost" scenario means the same thing for every model.
  - VOT comes out in real units without a hard-coded "× 60".

  The cost: reported coefficients are per original unit, so a coefficient per 100 minutes in 0.1.x becomes per minute. Likelihoods, probabilities and VOTs are unchanged. **[decide]**
- **Derived attributes are explicit columns.** ModeCanada's "time = ivt + ovt" becomes a real column in the canonical wide table (`time_train`, ...), declared as attribute `time`, next to `ivt` and `ovt`.

  Caveat: perturbing `time` does not change `ivt`. A model built on `ivt`/`ovt` will therefore not respond to a `time` scenario. Mitigation: spec v2 records which data columns a spec uses, and the perturbation engine warns when a perturbed column is not among a spec-built model's inputs. For models that aren't built from a spec, documentation is the only safeguard.

### 2.2 Units
Units are a small closed vocabulary:
- **Time:** `second`, `minute`, `hour`.
- **Cost:** a currency code (`CHF`, `CAD`, `GBP`, `USD`, `EUR`), optionally with a scale such as `"0.01 CHF"` if a dataset stores cents.

VOT is always reported as `<currency>/hour`. Conversion is done by DCMBench, never by the model. Unit-aware arithmetic stays this small; we don't take on a units library.

## 3. Datasets (`dcmbench.data`)

```python
ds = dcmbench.data.load("swissmetro")            # -> Dataset
ds.frame        # pd.DataFrame, canonical wide table (raw values, original units)
ds.problem      # ChoiceProblem
ds.info         # citation, license, source URL, version, sha256, row counts, codebook URL
ds.provenance   # list of applied steps (for the specification card)

sample = ds.filter("PURPOSE in (1, 3)", label="commuter and business trips")
lpmc = dcmbench.data.load("lpmc").subsample(5000, seed=42)
```

### 3.1 Behavior
- **Pinned, verified files.** Each dataset version maps to a file in `dcmbench-data` *at a fixed commit SHA*, never `main`, together with its sha256. Downloads are checked against the hash, and a mismatch raises an error. The hashes as of 2026-10-20 have been recorded for the first registry.
- **Default cache.** The cache is `~/.cache/dcmbench`, overridable with `DCMBENCH_DATA_DIR`. Files are downloaded once and work offline afterwards.
- **No hidden preprocessing.** Loading never drops rows silently. Filters and subsamples are explicit method calls that return a new `Dataset` and append a step to `provenance`. Each dataset's documented "standard" preparation is a named function, e.g. `ds.prepare("benchmark")`, so scripts read clearly.
- **Canonical tables are built in code, once.** For ModeCanada, the long-to-wide conversion lives in `dcmbench/data/datasets/modecanada.py`, with tests. It is not copied into scripts.
- **One module per dataset.** Each module contains the registry entry (files, hashes, citation), the canonical-table builder, the `ChoiceProblem`, and any named preparations. "Add a dataset" (R1) then means "add one module like these", and the guide documents the pattern.
- **Hosting.** Hosting stays on GitHub (`dcmbench-data`), pinned by commit. Zenodo, with a DOI, is a later option that changes only the registry URLs. **[decide]** whether that happens before release.

Reproducibility check: the 0.1.x scripts' LPMC sample was `sample(n=5000, random_state=42)` on the full table. Dummy creation and `dropna()` removed no rows, so `subsample(5000, seed=42)` must reproduce the same rows. A test pins this against the Phase 0 fixtures.

## 4. Protocols

The core protocol **is** the validation module's `FittedModel`. `dcmbench.protocols` re-exports it, and the validation module stays the definition.

```python
FittedModel           # predict_probabilities(data) -> DataFrame[alternatives]; n_params; null_loglikelihood(data); converged
ModelFitter           # Callable[[DataFrame], FittedModel]
```

### 4.1 The one change to `validation/`
`n_params` becomes `int | None`, where `None` means not applicable. `validation/metrics.likelihood_metrics` (the only consumer, via `validator.py`) then reports ρ̄² as NaN, flagged as not applicable, instead of failing on `total_ll - None`. Her tests are unaffected. A new test covers the `None` case.

### 4.2 Probability tables
`predict_probabilities` must return columns equal to `problem.alternatives`, in that order. Rows sum to 1, and unavailable alternatives get exactly 0. A shared checker, `check_probabilities(probs, problem, data)`, enforces this at every boundary and raises on violation. That's what replaces the silent fallbacks.

**[decide]** Should engines be *required* to apply availability, or should DCMBench mask and renormalize centrally? I recommend this split:
- DCMBench masks centrally for engines that cannot know about availability (sklearn).
- DCMBench raises if a model claiming to handle availability returns mass on an unavailable alternative.

### 4.3 Optional capabilities
Optional capabilities are detected with `isinstance` on runtime-checkable protocols. They are never required.

```python
class ProvidesCoefficients(Protocol):
    def coefficients(self) -> pd.DataFrame: ...       # name, estimate, robust s.e., fixed?

class ProvidesMarginalUtilities(Protocol):            # enables VOT Route A (Plan §5.2)
    def marginal_utilities(self, data, attributes: Sequence[str]) -> MarginalUtilities: ...

@dataclass
class MarginalUtilities:
    values: np.ndarray        # (n_obs, n_draws, n_alts, n_attributes): ∂V_ij / ∂x_ij in data units
    weights: np.ndarray       # (n_obs, n_draws): prior weights, rows sum to 1
    chosen_prob: np.ndarray | None = None   # (n_obs, n_draws): P(observed choice | draw), enables posteriors
```

One structure covers every model with a parameter story:

| Model | `n_draws` | `weights` | Notes |
|---|---|---|---|
| MNL, segmented MNL | 1 | 1 | Segment differences arise from interactions in V |
| Latent class | K classes | class-membership probabilities | |
| MXL | R draws | 1/R | `chosen_prob` gives Train-style posteriors (with panel products over a respondent's choices) |
| TasteNet | 1 | 1 | Per-person coefficients from the network |

**Route A VOT** is `values[..., time] / values[..., cost] × unit conversion`, weighted by `weights`, or by posterior weights ∝ `weights × chosen_prob` when asked. Defining Route A through marginal utilities, rather than "β_time/β_cost", handles nonlinear specs automatically. For example, the CUSTOM model's `log(1 + cost)` gives a VOT that varies by person. Spec-built models (Phase 3) get this capability for free, by differentiating the compiled expression. **Route B** (probability derivatives) needs nothing from the model.

## 5. Results

Each analysis returns a small result object holding a tidy DataFrame plus metadata (settings, provenance and route flags).

```python
@dataclass
class EvaluationResult:
    model_name: str
    problem: ChoiceProblem
    provenance: Provenance                 # dataset steps, spec hash, engine versions, seeds
    fit: FitResult | None                  # in-sample LL, ρ², ρ̄² (or n/a), K, converged
    validation: ValidationResult | None    # hphoebe's object, unchanged
    elasticities: ElasticityResult | None
    vot: VOTResult | None                  # includes route ("A-population", "A-posterior", "B")
    scenarios: ScenarioResult | None
    regularity: RegularityResult | None

    def to_card(self) -> dict: ...         # Phase 5; schema-validated
```

Every result object can be written to and read from JSON. Fields that don't apply are explicit nulls with a reason, never zeros.

## 6. Phase 1 scope

**Build now:**
- `dcmbench/problem.py`: `Attribute`, `ChoiceProblem`, validation, and the unit vocabulary with its conversions.
- `dcmbench/data/`:
  - registry, download, hash check, cache, `Dataset`, filters, subsample and provenance;
  - modules for Swissmetro, ModeCanada (long-to-wide) and LPMC, each with its `ChoiceProblem`, plus named benchmark preparations.
- `dcmbench/protocols.py`: re-exports, capability protocols, `MarginalUtilities`, `check_probabilities`.
- The `n_params: int | None` change in `validation/`.
- Result dataclasses and their JSON round-trip (empty shells; they are filled in Phase 2).
- **Tests:**
  - `ChoiceProblem` validation;
  - the hash check, using a local fake registry, so CI never needs the network;
  - the ModeCanada conversion, checked against 0.1.x numbers (N = 4,324 and mode shares);
  - the Swissmetro benchmark sample (N = 6,768) and the LPMC subsample (same rows as 0.1.x);
  - the probability checker;
  - hphoebe's 21 tests, unchanged.

**Not yet:**
- evaluation (Phase 2);
- engines and spec v2 (Phase 3);
- deleting the old modules, which keep importing until Phase 3 so nothing is half-broken in between.

## 7. Decisions needed

| # | Question | Recommendation |
|---|---|---|
| 1 | Keep data in original units and move scaling into specs, so coefficients become per original unit? | Yes |
| 2 | Add Chicago as a fourth dataset? | Only if its source, license and codebook can be found. Otherwise leave it out. |
| 3 | Availability: mask centrally for sklearn-type engines, and raise if a model that claims to handle availability violates it? | Yes |
| 4 | Data hosting: GitHub pinned by commit now, Zenodo DOI before release? | GitHub now. Decide on Zenodo at Phase 6. |
| 5 | Biogeme as an optional extra (`dcmbench[biogeme]`) rather than core? | Yes. The core then needs only numpy, pandas, scipy and scikit-learn. |
| 6 | Route A defined through marginal utilities (§4.3), not coefficient ratios? | Yes |
| 7 | Dataset names: `swissmetro`, `modecanada`, `lpmc` (`ltds` kept as an alias)? | Yes. LPMC is the name used in the literature. |
