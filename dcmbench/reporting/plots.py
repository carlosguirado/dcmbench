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


def plot_fit_vs_prediction(cards: Sequence[dict], ax=None):
    """In-sample vs out-of-sample log-likelihood per observation, one point per card."""
    figure, ax = _axes(ax)
    datasets = sorted({c["data"]["dataset"] for c in cards})
    models = sorted({c["model"]["name"] for c in cards})
    markers = dict(zip(models, "osD^vP*Xh<>p"))
    for card in cards:
        if not card["prediction"] or "avg_ll" not in card["prediction"]["metrics"]:
            continue
        x = card["fit"]["loglikelihood"] / card["fit"]["n_obs"]
        y = card["prediction"]["metrics"]["avg_ll"]["estimate"]
        color = f"C{datasets.index(card['data']['dataset'])}"
        ax.scatter(x, y, color=color, marker=markers.get(card["model"]["name"], "o"), s=40)
    lows = [min(c["fit"]["loglikelihood"] / c["fit"]["n_obs"], c["prediction"]["metrics"]["avg_ll"]["estimate"])
            for c in cards if c["prediction"] and "avg_ll" in c["prediction"]["metrics"]]
    if lows:
        low = min(lows)
        ax.plot([low, 0], [low, 0], color="grey", linestyle="--", linewidth=0.8, label="OOS = IS")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=f"C{i}", marker="o", linestyle="", label=d) for i, d in enumerate(datasets)]
    handles += [Line2D([], [], color="black", marker=markers[m], linestyle="", label=m) for m in models]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("in-sample log-likelihood per observation")
    ax.set_ylabel("out-of-sample log-likelihood per observation")
    return figure
