"""Seeded split generators for k-fold CV and bootstrap validation."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def _as_generator(random_state: Optional[int]) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)


def iter_kfold_indices(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    group_column: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_indices, test_indices) for k-fold cross-validation.

    When ``group_column`` is set, entire groups stay in the same fold
    (GroupKFold). Otherwise uses shuffled KFold at the row level.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(data) < n_splits:
        raise ValueError(
            f"Not enough observations ({len(data)}) for {n_splits}-fold CV"
        )

    n = len(data)
    positions = np.arange(n)

    if group_column is not None:
        if group_column not in data.columns:
            raise ValueError(f"group_column '{group_column}' not found in data")
        groups = data[group_column].to_numpy()
        n_groups = pd.Series(groups).nunique()
        if n_groups < n_splits:
            raise ValueError(
                f"Not enough groups ({n_groups}) for {n_splits}-fold CV "
                f"with group_column='{group_column}'"
            )
        splitter = GroupKFold(n_splits=n_splits)
        # GroupKFold is deterministic given group order; shuffle groups via
        # a stable remapping seeded by random_state when provided.
        if random_state is not None:
            rng = _as_generator(random_state)
            unique_groups = pd.unique(groups)
            shuffled = unique_groups.copy()
            rng.shuffle(shuffled)
            remap = {g: i for i, g in enumerate(shuffled)}
            groups_for_split = np.array([remap[g] for g in groups])
        else:
            groups_for_split = groups
        for train_idx, test_idx in splitter.split(positions, groups=groups_for_split):
            yield train_idx, test_idx
    else:
        splitter = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for train_idx, test_idx in splitter.split(positions):
            yield train_idx, test_idx


def iter_bootstrap_indices(
    data: pd.DataFrame,
    n_bootstrap: int = 30,
    random_state: Optional[int] = None,
    group_column: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_indices, oob_indices) for prediction-error bootstrap.

    Observation-level: sample ``n`` rows with replacement; OOB = never drawn.
    Group-level: sample unique groups with replacement; train rows are all
    observations from drawn groups (with multiplicity); OOB = groups never drawn.

    Draws with an empty OOB set are skipped.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    rng = _as_generator(random_state)
    n = len(data)

    if group_column is not None:
        if group_column not in data.columns:
            raise ValueError(f"group_column '{group_column}' not found in data")
        groups = data[group_column].to_numpy()
        unique_groups = pd.unique(groups)
        n_groups = len(unique_groups)
        group_to_rows = {
            g: np.where(groups == g)[0] for g in unique_groups
        }

        produced = 0
        attempts = 0
        max_attempts = n_bootstrap * 20
        while produced < n_bootstrap and attempts < max_attempts:
            attempts += 1
            drawn = rng.choice(unique_groups, size=n_groups, replace=True)
            drawn_set = set(drawn.tolist())
            oob_groups = [g for g in unique_groups if g not in drawn_set]
            if not oob_groups:
                continue
            train_parts = [group_to_rows[g] for g in drawn]
            train_idx = np.concatenate(train_parts)
            oob_idx = np.concatenate([group_to_rows[g] for g in oob_groups])
            yield train_idx, oob_idx
            produced += 1
        return

    produced = 0
    attempts = 0
    max_attempts = n_bootstrap * 20
    while produced < n_bootstrap and attempts < max_attempts:
        attempts += 1
        train_idx = rng.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[train_idx] = False
        oob_idx = np.where(oob_mask)[0]
        if len(oob_idx) == 0:
            continue
        yield train_idx, oob_idx
        produced += 1
