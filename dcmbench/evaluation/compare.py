"""Evaluate several models on one dataset under identical splits, metrics and analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import pandas as pd

from dcmbench.evaluation.run import CrossValidation, evaluate
from dcmbench.results import EvaluationResult


@dataclass
class Comparison:
    """Results of ``compare()``: one ``EvaluationResult`` per model."""

    dataset: str
    results: Dict[str, EvaluationResult]

    def __getitem__(self, name: str) -> EvaluationResult:
        return self.results[name]

    def cards(self) -> Dict[str, Dict[str, Any]]:
        from dcmbench.reporting.card import build_card

        return {name: build_card(result) for name, result in self.results.items()}

    def table(self) -> pd.DataFrame:
        from dcmbench.reporting.tables import compare_cards

        return compare_cards(self.cards())

    def save(self, directory: Union[str, Path]) -> Path:
        """Write one card (JSON and Markdown) per model and the comparison table (CSV)."""
        from dcmbench.reporting.card import render_markdown, save_card

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cards = self.cards()
        for name, card in cards.items():
            stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{self.dataset}_{name}")
            save_card(card, directory / f"{stem}.card.json")
            (directory / f"{stem}.card.md").write_text(render_markdown(card))
        from dcmbench.reporting.tables import compare_cards

        compare_cards(cards).to_csv(directory / f"{self.dataset}_comparison.csv")
        return directory


def compare(models: Mapping[str, Any], dataset, *, protocol=None, **evaluate_options) -> Comparison:
    """
    Evaluate each model or fitter in ``models`` on ``dataset`` with the same
    protocol (default: 5-fold respondent-grouped cross-validation), scenarios
    and options. The protocol's seed fixes the folds, so every model is
    trained and scored on identical splits.
    """
    if not models:
        raise ValueError("give at least one model")
    protocol = CrossValidation() if protocol is None else protocol
    results = {name: evaluate(model, dataset, name=name, protocol=protocol, **evaluate_options)
               for name, model in models.items()}
    return Comparison(dataset.name, results)
