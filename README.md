# Discrete Choice Model Benchmarking (DCMBench)

A comprehensive Python package for benchmarking, analyzing, and validating discrete choice models for transportation mode choice analysis. DCMBench provides a unified framework for comparing different model specifications, conducting sensitivity analysis, and visualizing results.

## Installation

You can install DCMBench using pip:

```bash
pip install dcmbench
```

## What to use when

| Need | Use |
|------|-----|
| In-sample fit / compare models on estimation data | `SimpleBenchmarker` |
| Internal validation (k-fold CV / prediction-error bootstrap) | `dcmbench.validation` |
| Parameter standard errors via bootstrap | Not in DCMBench (see Apollo-style tools) |

## Key Features

### Model Estimation and Benchmarking

- **Multiple Model Types**: Support for Multinomial Logit (MNL), Nested Logit (NL), and Mixed Logit (ML) models
- **Standardized Metrics**: Compare models using log-likelihood, rho-squared, prediction accuracy, and market share
- **Internal validation**: K-fold cross-validation and prediction-error bootstrap (OOB + 0.632)
- **Visualization**: Generate comparative plots showing model performance across different metrics

```python
from dcmbench.model_benchmarker import SimpleBenchmarker
from dcmbench.datasets import fetch_data

# Load dataset (automatically downloads if not in local cache)
data = fetch_data("swissmetro_dataset")

# Define models and run benchmark (in-sample scoring of already-fitted models)
benchmarker = SimpleBenchmarker()
benchmarker.register_model(adapter, "Model Name")
results = benchmarker.run_benchmark(data, choice_column="CHOICE")
benchmarker.print_comparison()
```

### Internal Validation (Cross-Validation and Bootstrap)

`dcmbench.validation` re-estimates the model on each training fold or bootstrap sample and scores held-out data. This is separate from in-sample `SimpleBenchmarker` metrics.

```python
from dcmbench.datasets import fetch_data
from dcmbench.model_specifications import fetch_model_spec
from dcmbench.utils.model_extensions import prepare_swissmetro_data_standard
from dcmbench.validation import InternalValidator, fitter_from_spec

data = prepare_swissmetro_data_standard(fetch_data("swissmetro_dataset"))
spec = fetch_model_spec("mode_choice/mnl_swissmetro.json")
fitter = fitter_from_spec(spec, choice_column="CHOICE")

validator = InternalValidator(
    fitter,
    choice_column="CHOICE",
    choice_mapping={1: 1, 2: 2, 3: 3},
    group_column="ID",       # recommended for panel / repeated choices
    random_state=42,
)

cv = validator.cross_validate(data, n_splits=5)
print(cv.summary())

boot = validator.bootstrap(data, n_bootstrap=30)
print(boot.summary())  # includes *_oob and *_632 prediction metrics
```

See `tutorials/internal_validation_swissmetro.py` for a runnable example.

**Notes:**
- Each fold/draw is a full estimation (can be slow for mixed logit).
- Bootstrap here estimates **out-of-sample predictive fit**, not parameter covariance.
- `choice_mapping` is `{CHOICE column code -> predict_probabilities() column name}`.
  These two label systems aren't guaranteed to match (e.g. `CHOICE` could store
  strings while probability columns are integers); `{1: 1, 2: 2, 3: 3}` above is
  an identity map because Swissmetro's `CHOICE` codes and
  `UniversalBiogemeAdapter`'s probability-column names both happen to use the
  same numeric alternative-ID convention (1/2/3 = Train/Swissmetro/Car), not
  because the mapping is inherently redundant.
- MNL/NL are the supported paths for `fitter_from_spec`/`BiogemeSpecFitter`.
  **Mixed logit (MXL) specs are not supported through the JSON-spec path**:
  `build_model_from_spec` does not wire `random_parameters` into the utility
  formulas (no `bioDraws` are created), so specs with `model_type: "MXL"` and a
  non-empty `random_parameters` raise a clear error, and specs with an empty
  `random_parameters` are silently estimated as plain MNL (with a warning).
  **To validate a genuine mixed logit model, use `FunctionModelFitter`** to wrap
  a hand-built Biogeme model with explicit `bioDraws` (`InternalValidator` is
  model-agnostic and works with any `build_fn`/`predict_fn` pair -- the same
  `build_*`/`predict_*` pattern used for MXL in `dcm-internal-validation`):

  ```python
  from dcmbench.validation import FunctionModelFitter, InternalValidator

  fitter = FunctionModelFitter(
      build_fn=build_mxl,               # (train_df) -> Biogeme estimation results
      predict_fn=predict_mxl,           # (results, data) -> probabilities DataFrame
      null_loglikelihood_fn=null_loglikelihood_mxl,  # (results, data) -> float
      choice_column="CHOICE",
  )
  validator = InternalValidator(fitter, choice_column="CHOICE", choice_mapping={1: 1, 2: 2, 3: 3})
  cv = validator.cross_validate(data, n_splits=5)
  ```

  See `tutorials/internal_validation_mxl_swissmetro.py` for a full runnable
  example (Swissmetro MXL with a normally-distributed time coefficient).

### Advanced Analysis Capabilities

- **Sensitivity Analysis**: Evaluate how model predictions change with variations in key variables
  - Simulate changes in travel times, costs, and other attributes
  - Generate plots showing the evolution of market shares under different scenarios

- **Individual-Level Parameters**: For Mixed Logit models, calculate and visualize:
  - Individual-specific parameter distributions using Bayesian approaches
  - Value of Time (VOT) distributions across the population
  - Heterogeneity in preference structures

- **Model Calibration**: Automatically calibrate Alternative Specific Constants (ASCs) to match observed market shares

### Visualization and Reporting

- **Market Share Analysis**: Compare predicted vs. observed mode shares
- **Performance Plots**: Visualize how different models perform across multiple datasets
- **Parameter Distributions**: Plot distributions of random parameters and derived metrics like VOT
- **Sensitivity Curves**: Show how predicted mode shares change with variations in key variables

## Supported Datasets

- **Swissmetro** (`swissmetro_dataset`): Swiss inter-city travel mode choice
- **London Transport** (`ltds_dataset`): London Travel Demand Survey with urban mode choices
- **ModeCanada** (`modecanada_dataset`): Canadian inter-city travel dataset

Datasets are automatically downloaded from the [dcmbench-datasets](https://github.com/carlosguirado/dcmbench-datasets) repository on first use and cached locally:

```python
from dcmbench.datasets import fetch_data

# Use default cache location (~/.dcmbench/datasets)
data = fetch_data("swissmetro_dataset")

# Specify custom cache location
data = fetch_data("swissmetro_dataset", local_cache_dir="/path/to/cache")

# Get features and target separately
X, y = fetch_data("swissmetro_dataset", return_X_y=True)
```

## Example Applications

### Benchmarking Multiple Models

The package includes tools to benchmark multiple model types across different datasets:

```python
# Run benchmark_all_models.py to compare models across datasets
python benchmark_all_models.py
```

This generates comparative visualizations showing how different model types perform across datasets, plotting metrics like choice accuracy and market share accuracy against model fit.

### Sensitivity Analysis

Analyze how changes in key variables affect predicted mode shares:

```python
# Run sensitivity analysis on ModeCanada models
python sensitivity_analysis.py
```

This creates plots showing the evolution of mode shares as you modify variables like:
- Travel costs for different modes
- Travel times
- Service frequencies

### Individual Parameter Analysis

For Mixed Logit models, analyze individual-level parameters and VOT:

```python
# Generate individual parameter distributions for ModeCanada
python plot_individual_parameters_canada.py
```

This calculates individual-specific parameters using Bayesian conditioning and produces:
- Distributions of time and cost parameters
- Value of Time (VOT) distributions
- Summary statistics for preference heterogeneity

## Requirements

- Python >=3.8
- NumPy >=1.19.0
- Pandas >=1.2.0
- Biogeme >=3.2.0
- scikit-learn >=0.24.0
- Matplotlib >=3.3.0
- Requests >=2.25.0
- SciPy (for statistical functions)
- Seaborn (for advanced visualizations)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Adding New Datasets

To add a new dataset to the DCMBench ecosystem:

1. Fork the [dcmbench-datasets](https://github.com/carlosguirado/dcmbench-datasets) repository
2. Add your dataset following the structure guidelines in the repository's CONTRIBUTING.md file
3. Submit a pull request to the dcmbench-datasets repository
4. Update the metadata.json file in the main DCMBench package to include your dataset information

This design allows the package to remain lightweight while providing access to a growing collection of transportation mode choice datasets.
