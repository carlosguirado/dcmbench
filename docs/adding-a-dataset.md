# Adding a dataset

## Your own data, in your own code

Describe the table with a `ChoiceProblem`, then wrap it:

```python
import dcmbench.data
from dcmbench import Attribute, ChoiceProblem

problem = ChoiceProblem(
    name="my survey",
    alternatives=("bus", "rail", "car"),
    choice_column="mode",
    choice_codes={"b": "bus", "r": "rail", "c": "car"},
    availability={"car": "has_car"},                 # columns of 0/1; omitted = always available
    attributes={
        "time": Attribute({"bus": "t_bus", "rail": "t_rail", "car": "t_car"}, unit="minute", kind="time"),
        "cost": Attribute({"bus": "c_bus", "rail": "c_rail", "car": "c_car"}, unit="USD", kind="cost"),
    },
    group_column="respondent",                       # keeps each respondent in one fold
)
dataset = dcmbench.data.from_frame(table, problem, name="my survey", citation="Me et al. (2026)")
```

`from_frame` validates the table against the problem. Checks include:
- every choice code is known;
- the chosen alternative is available;
- every attribute column exists and is numeric.

It also records a content hash, so results trace back to the exact data.
Units are `minute` or `hour` for time and any currency code for cost. VOT is
reported in that currency per hour.

## A built-in dataset (contributing to DCMBench)

A built-in dataset is one module in `dcmbench/data/datasets/`, defining
`PROBLEM`, a `build` function and an `ENTRY`. Use `swissmetro.py` as the
template:

1. **Host the file.** Put a CSV (gzip is fine) at a stable, versioned URL. The built-in datasets live in the `dcmbench-data` repository, pinned to a commit.
2. **Pin it.** `DataFile(url=..., sha256=...)`. Downloads are verified, so a changed file fails loudly.
3. **Describe it.** Write the `ChoiceProblem`, with units and the respondent column.
4. **Build the canonical table.** `build(raw) -> (frame, steps)` makes the minimal, documented changes, e.g. dropping rows with no recorded choice. Each one is recorded as a `Step` with counts before and after.
5. **Name sample definitions.** If there is a standard estimation sample, add it to `preparations` (e.g. `{"benchmark": benchmark_sample}`).
6. **Credit the source.** Fill in `citation`, `source_url` and `license`, giving the terms under which the data may be used and redistributed. Check them with the original source before submitting.
7. **Register it.** Add the module to `_BUILTIN_MODULES` in `dcmbench/data/registry.py`, and add a network test with the expected sample size and shares (see `tests/test_datasets_network.py`).

A dataset can also be registered at runtime without changing DCMBench:
`dcmbench.data.register(DatasetEntry(...))`.
