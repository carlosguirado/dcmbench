"""
Reporting: specification cards (prototype), comparison tables and figures.

    from dcmbench.evaluation import compare
    comparison = compare({"MNL": mnl_fitter, "TasteNet": tastenet_fitter}, dataset)
    comparison.table()                 # side by side
    comparison.save("results/")       # one card (JSON + Markdown) per model, plus the table
"""

from dcmbench.reporting.card import CARD_VERSION, build_card, load_card, render_markdown, save_card, validate_card
from dcmbench.reporting.tables import compare_cards

__all__ = ["CARD_VERSION", "build_card", "compare_cards", "load_card", "render_markdown", "save_card", "validate_card"]
