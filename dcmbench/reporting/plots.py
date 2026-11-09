"""
Comparison figures (matplotlib).

- ``plot_vot``: VOT distributions of several models for one alternative.
  Point values (MNL) are drawn as spikes, discrete distributions (segments,
  latent classes) as weighted spikes, continuous ones (mixed logit,
  TasteNet) as weighted densities. Needs ``evaluate(..., keep_vot_samples=True)``.
- ``plot_sensitivity``: predicted shares against an attribute multiplier.
  Needs ``evaluate(..., sensitivity=(attribute, alternatives, multipliers))``.
- ``plot_fit_vs_prediction``: in-sample against out-of-sample log-likelihood
  per observation, from cards.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from dcmbench.results import EvaluationResult

# Up to this many distinct values, a VOT distribution is drawn as weighted spikes.
DISCRETE_LIMIT = 10


def _axes(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4))
        return figure, ax
    return ax.figure, ax


def _vot_samples(name, result, alternative):
    if "vot" not in result.analyses or "samples" not in result["vot"].tables:
        raise ValueError(f"{name}: evaluate with keep_vot_samples=True to plot VOT distributions")
    samples = result["vot"].tables["samples"]
    samples = samples[samples.alternative == alternative]
    values = samples["vot"].to_numpy(dtype=float)
    weights = samples["weight"].to_numpy(dtype=float)
    return values, (weights / weights.sum() if len(weights) else weights)


def plot_vot(results: Mapping[str, EvaluationResult], alternative: str, ax=None,
             limits: Optional[Tuple[float, float]] = None, bins: int = 60):
    """
    VOT distribution of each model for ``alternative`` (currency per hour).

    Continuous distributions are drawn as densities (left axis); point and
    discrete ones (MNL, segments, latent classes) as spikes whose height is
    the population share (right axis). The default range runs from the
    median, across continuous models, of their 1st percentiles to the median
    of their 99th percentiles, widened to include spikes within ten times
    that range and always including 0. One extreme model therefore cannot set
    the axis; every legend entry reports the share of that model's
    distribution outside the range.
    """
    from matplotlib.lines import Line2D

    from dcmbench.evaluation.vot import weighted_quantiles

    figure, ax = _axes(ax)
    samples = {name: _vot_samples(name, result, alternative) for name, result in results.items()}
    samples = {name: s for name, s in samples.items() if len(s[0])}
    if not samples:
        raise ValueError(f"no model has VOT values for {alternative!r}")
    discrete = {name: len(np.unique(np.round(v, 6))) <= DISCRETE_LIMIT for name, (v, _) in samples.items()}

    if limits is None:
        continuous = [n for n in samples if not discrete[n]] or list(samples)
        quantiles = np.array([weighted_quantiles(*samples[n], [0.01, 0.99]) for n in continuous])
        low, high = float(np.median(quantiles[:, 0])), float(np.median(quantiles[:, 1]))
        span = max(high - low, 1e-9)
        spikes = [v for n in samples if discrete[n] for v in np.unique(samples[n][0])]
        reachable = [v for v in spikes if low - 10 * span <= v <= high + 10 * span]
        low, high = min([low, 0.0] + reachable), max([high, 0.0] + reachable)
        pad = 0.03 * (high - low)
        limits = (low - pad, high + pad)
    low, high = limits

    share_axis = ax.twinx() if any(discrete.values()) else None
    handles, currency = [], None
    for index, (name, (values, weights)) in enumerate(samples.items()):
        analysis = results[name]["vot"]
        currency = analysis.settings["currency_per_hour"]
        color = f"C{index}"
        inside = (values >= low) & (values <= high)
        outside = float(weights[~inside].sum())
        label = f"{name} ({analysis.settings['route']})" + (f", {outside:.0%} outside" if outside >= 0.005 else "")
        if discrete[name]:
            totals = {}
            for v, w in zip(np.round(values[inside], 6), weights[inside]):
                totals[v] = totals.get(v, 0.0) + w
            xs, ys = np.array(list(totals), dtype=float), np.array(list(totals.values()), dtype=float)
            share_axis.vlines(xs, 0, ys, color=color, linewidth=2.5)
            share_axis.plot(xs, ys, "o", color=color)
            handles.append(Line2D([], [], color=color, linewidth=2.5, marker="o", label=label))
        else:
            if inside.any():
                ax.hist(values[inside], bins=bins, range=(low, high), weights=weights[inside],
                        density=True, histtype="step", linewidth=1.8, color=color)
            handles.append(Line2D([], [], color=color, linewidth=1.8, label=label))
    ax.set_xlim(low, high)
    ax.set_xlabel(f"value of time ({currency}/hour), {alternative}")
    ax.set_ylabel("density (continuous distributions)")
    if share_axis is not None:
        share_axis.set_ylim(0, 1.05)
        share_axis.set_ylabel("population share (point and discrete distributions)")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title(f"VOT distributions: {alternative}", fontsize=10)
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.12 if share_axis else 1.01, 1.0))
    return figure


def plot_sensitivity(results: Mapping[str, EvaluationResult], alternative: str, ax=None):
    """Predicted share of ``alternative`` along the sensitivity curve of each model."""
    figure, ax = _axes(ax)
    settings = None
    for name, result in results.items():
        if "sensitivity" not in result.analyses:
            raise ValueError(f"{name}: evaluate with sensitivity=(attribute, alternatives, multipliers)")
        analysis = result["sensitivity"]
        settings = analysis.settings
        curve = analysis.summary[analysis.summary.alternative == alternative].sort_values("multiplier")
        ax.plot(curve["multiplier"], curve["share"], marker="o", markersize=3, label=name)
    targets = ", ".join(settings["alternatives"]) if settings and settings["alternatives"] else "all alternatives"
    ax.set_xlabel(f"{settings['attribute']} multiplier ({targets})" if settings else "multiplier")
    ax.set_ylabel(f"predicted share of {alternative}")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=8)
    return figure


# Categorical colours in fixed order (validated default palette); markers per dataset.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DATASET_MARKERS = "os^Dv"


def plot_fit_vs_prediction(cards: Sequence[dict], ax=None, groups: Optional[Mapping[str, str]] = None,
                           label_gap: float = 0.1, dataset_labels: Optional[Mapping[str, str]] = None):
    """
    In-sample vs out-of-sample log-likelihood per observation, one point per card.

    Shape encodes the dataset. Colour encodes ``groups[model]`` (e.g. model
    family) when given, else the model itself; at most eight colours are used,
    so with more models pass ``groups``. Points whose out-of-sample value falls
    more than ``label_gap`` below the in-sample one are labelled with the model
    name (the generalization failures the figure is about). ``dataset_labels``
    maps dataset names to display names.
    """
    from matplotlib.lines import Line2D

    figure, ax = _axes(ax)
    points = [c for c in cards if c["prediction"] and "avg_ll" in c["prediction"]["metrics"]]
    datasets = sorted({c["data"]["dataset"] for c in points})
    key = (lambda m: groups.get(m, m)) if groups else (lambda m: m)
    categories = list(dict.fromkeys(key(c["model"]["name"]) for c in points))
    if groups:
        order = list(dict.fromkeys(groups.values()))
        categories = [g for g in order if g in categories] + [g for g in categories if g not in order]
    if len(categories) > len(CATEGORICAL):
        raise ValueError(f"{len(categories)} colour categories; at most {len(CATEGORICAL)} are distinguishable. "
                         "Pass groups={model: family} to colour by family.")
    colors = dict(zip(categories, CATEGORICAL))
    markers = dict(zip(datasets, DATASET_MARKERS))

    shown_name = (lambda d: dataset_labels.get(d, d)) if dataset_labels else (lambda d: d)
    values = [(c["model"]["name"], c["data"]["dataset"], c["fit"]["loglikelihood"] / c["fit"]["n_obs"],
               c["prediction"]["metrics"]["avg_ll"]["estimate"]) for c in points]
    if not values:
        raise ValueError("no card has an out-of-sample avg_ll")
    xs = np.array([v[2] for v in values])
    ys = np.array([v[3] for v in values])
    # Vertical floor: points far below the cloud are drawn at the edge with their value.
    x_span = max(xs.max() - xs.min(), 1e-9)
    floor = min(xs.min(), np.sort(ys)[min(1, len(ys) - 1)]) - 0.35 * x_span
    labels = []
    for (model, dataset, x, y) in values:
        shown = max(y, floor)
        ax.scatter(x, shown, color=colors[key(model)], marker=markers[dataset], s=46, edgecolors="white",
                   linewidths=1.0, zorder=3)
        if y < floor:
            ax.annotate("", (x, floor), xytext=(x, floor + 0.04 * x_span),
                        arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.8))
        if x - y > label_gap:
            labels.append([shown, x, f"{model}, {shown_name(dataset)}" + (f" ({y:.2f})" if y < floor else "")])
    low = min(xs.min(), floor)
    ax.plot([low, 0], [low, 0], color="#52514e", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_ylim(floor - 0.05 * x_span, max(ys.max(), xs.max()) + 0.08 * x_span)
    ax.set_xlim(xs.min() - 0.05 * x_span, xs.max() + 0.55 * x_span)
    # Labels to the right of the cloud, spread vertically so they never overlap, with leader lines.
    if labels:
        labels.sort(key=lambda item: -item[0])
        gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        positions = []
        for y, _, _ in labels:
            positions.append(min(y, positions[-1] - gap) if positions else y)
        label_x = xs.max() + 0.12 * x_span
        for (y, x, text), ly in zip(labels, positions):
            ax.annotate(text, (x, y), xytext=(label_x, ly), textcoords="data", fontsize=7, color="#52514e",
                        va="center", arrowprops=dict(arrowstyle="-", color="#b5b4ae", linewidth=0.6))
    ax.grid(color="#e6e5e1", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[g], marker="o", linestyle="", markersize=7, label=g) for g in categories]
    handles += [Line2D([], [], color="#52514e", marker=markers[d], linestyle="", markersize=7, label=shown_name(d))
                for d in datasets]
    handles.append(Line2D([], [], color="#52514e", linestyle="--", linewidth=0.8, label="out-of-sample = in-sample"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    ax.set_xlabel("in-sample log-likelihood per observation")
    ax.set_ylabel("out-of-sample log-likelihood per observation")
    return figure
