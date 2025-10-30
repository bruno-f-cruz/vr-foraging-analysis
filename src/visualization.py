from matplotlib import pyplot as plt
import numpy as np
from aind_behavior_vr_foraging import task_logic as vrf_task
from typing import Optional, Literal, Any, Callable
from .dataset import SessionDataset
import pandas as pd
import logging
from functools import partial
from contextlib import contextmanager


from itertools import cycle

logger = logging.getLogger(__name__)

patch_index_colormap = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]


def get_color_from_site(site_label: str, patch_idx: int) -> str:
    if site_label == vrf_task.VirtualSiteLabels.REWARDSITE:
        base_color = patch_index_colormap[patch_idx % len(patch_index_colormap)]
    elif site_label == vrf_task.VirtualSiteLabels.INTERPATCH:
        base_color = "#A9A9A9"
    elif site_label == vrf_task.VirtualSiteLabels.INTERSITE:
        base_color = "#4C4C4C"
    else:
        raise ValueError(f"Unknown site label: {site_label}")
    return base_color


def plot_ethogram(
    dataset: SessionDataset,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot an ethogram of the session, showing the different sites, velocity, and events.
    Parameters
    ----------
    dataset : SessionDataset
        The session dataset to plot.
    t_start : Optional[float], optional
        The start time of the window to plot. If None, uses the start of the session., by default None
    t_end : Optional[float], optional
        The end time of the window to plot. If None, uses the end of the session., by default None
    ax : Optional[plt.Axes], optional
        The axes to plot on. If None, creates a new figure and axes., by default None
    **kwargs
        Additional keyword arguments to pass to plt.subplots if ax is None.
    Returns
    -------
    tuple[plt.Axes, plt.Axes]
        The axes with the ethogram and the axes with the events, respectively.
    """
    window_start = dataset.sites["t_start"].iloc[0] if t_start is None else t_start
    window_end = dataset.sites["t_end"].iloc[-1] if t_end is None else t_end

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

    mask = (dataset.sites["t_end"] >= window_start) & (
        dataset.sites["t_start"] <= window_end
    )
    for i, row in dataset.sites[mask].iterrows():
        color = get_color_from_site(row["site_label"], row["patch_idx"])
        ax.axvspan(
            row["t_start"],
            row["t_end"],
            color=color,
            alpha=0.6,
            label=row["plot_label"],
        )
    ax.plot(
        dataset.processed_streams.position_velocity["velocity"],
        color="k",
        label="Velocity",
        lw=3,
    )

    trials = dataset.trials
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.scatter(
        trials["choice_time"],
        np.ones_like(trials["choice_time"]) * 0.5,
        color="C3",
        label="Choices",
        marker="|",
        s=200,
        lw=2,
    )
    ax2.scatter(
        trials["reward_time"],
        np.ones_like(trials["reward_time"]) * 0.5,
        color="blue",
        label="Rewards",
        marker="|",
        s=100,
    )
    ax2.scatter(
        dataset.processed_streams.lick_onsets,
        np.ones_like(dataset.processed_streams.lick_onsets) * 0.6,
        color="green",
        label="Licks",
        marker="|",
        s=50,
    )
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Velocity (cm/s)")
    ax2.set_ylabel("Events")
    ax.set_xlim(window_start, window_end)

    # legend 1
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(-0.05, 1), loc="upper right"
    )

    # legend 2
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax, ax2


def plot_aligned_to(
    timestamps: np.ndarray | pd.Series | pd.DataFrame,
    timeseries: pd.Series | pd.DataFrame,
    *,
    event_window: tuple[float, float] = (-1, 1),
    ax: Optional[plt.Axes] = None,
    plot_func: Literal["scatter", "plot"] = "plot",
    **kwargs,
) -> tuple[plt.Axes, list[pd.Series | pd.DataFrame]]:
    _ax_passed = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

    plot_method = getattr(ax, plot_func)

    # Sanitize timestamps input
    if isinstance(timestamps, np.ndarray):
        assert timestamps.ndim == 1, "Timestamps array must be one-dimensional"
    elif isinstance(timestamps, (pd.Series, pd.DataFrame)):
        timestamps = timestamps.index.to_numpy()

    plot_kwargs = kwargs.pop("plot_kwargs", {})
    logger.debug(f"Plotting with kwargs: {plot_kwargs}")

    snippets = []
    for ts in timestamps:
        _win = np.array(event_window) + ts
        samples_in_window = timeseries.index[
            (timeseries.index >= _win[0]) & (timeseries.index <= _win[1])
        ]
        snippet = timeseries.loc[samples_in_window]
        snippets.append(snippet)
        plot_method(
            samples_in_window - ts,
            snippet,
            **plot_kwargs,
        )
    if not _ax_passed:
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("Value")
        ax.set_xlim(event_window)
        ax.axvline(0, color="k", ls="--", lw=1)

    return ax, snippets


def _get_cycle_cmap(n: int) -> cycle:
    colormap_obj = plt.get_cmap("tab10")
    cmap = [colormap_obj(i / (n - 1)) for i in range(n)]
    return cycle(cmap)


def _get_default_plot_kwargs(cmap: cycle) -> dict[str, Any]:
    return {
        "color": next(cmap),
        "alpha": 0.1,
        "linewidth": 1,
    }


def plot_aligned_to_grouped_by(
    timestamp_df: pd.DataFrame,
    timeseries: pd.Series,
    by: list[Any] | None = None,
    timestamp_column: str | None = None,
    plot_kwargs: Optional[dict[tuple[Any, ...], dict[str, Any]]] = None,
    *,
    agg_plot_kwarg_modifier: dict[str, Any] = {"alpha": 1, "linewidth": 2},
    agg_spread_kwarg_modifier: dict[str, Any] = {"alpha": 0.1, "linewidth": 0},
    event_window: tuple[float, float] = (-1, 1),
    ax: Optional[plt.Axes] = None,
    bin_width: float = 0.025,
    agg_fnc: Callable[[np.ndarray], np.ndarray] = partial(np.nanmean, axis=1),
    agg_spread_fnc: Callable[[np.ndarray], np.ndarray] = partial(
        np.nanpercentile, q=[2.5, 97.5], axis=1
    ),
    **kwargs,
) -> tuple[plt.Axes, dict[Any, pd.DataFrame]]:
    _ax_passed = ax is not None

    _anonymous_cmap = _get_cycle_cmap(10)

    by = by or []
    plot_kwargs = plot_kwargs or {}

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

    _summary_data = {}
    for _tup, df in timestamp_df.groupby(by):
        if timestamp_column is not None:
            timestamps = df[timestamp_column].to_numpy()
        else:
            timestamps = df.index.to_numpy()

        if _tup not in plot_kwargs:
            logging.warning(
                f"No plot_kwargs specified for group {_tup}, using defaults."
            )
            _these_plot_kwargs = _get_default_plot_kwargs(_anonymous_cmap)
        else:
            _these_plot_kwargs = plot_kwargs[_tup]

        ax, data = plot_aligned_to(
            timestamps,
            timeseries,
            event_window=event_window,
            plot_kwargs=_these_plot_kwargs,
            plot_func="plot",
            ax=ax,
        )
        # normalize each snippet to event time
        for d, onset in zip(data, timestamps):
            d.index = d.index - onset
        new_index = pd.Index(
            np.arange(
                min(s.index.min() for s in data),
                max(s.index.max() for s in data),
                bin_width,
            ),
        )

        binned_data = np.full((len(new_index), len(data)), np.nan)

        for i, s in enumerate(data):
            binned_data[:, i] = np.interp(new_index, s.index, s.values)

        binned_mean = agg_fnc(binned_data)
        binned_spread = agg_spread_fnc(binned_data)
        ax.plot(
            new_index,
            binned_mean,
            label=", ".join(f"{col}={val}" for col, val in zip(by, _tup))
            if by
            else str(_tup),
            **{**_these_plot_kwargs, **agg_plot_kwarg_modifier},
        )
        ax.fill_between(
            new_index,
            binned_spread[0],
            binned_spread[1],
            **{**_these_plot_kwargs, **agg_spread_kwarg_modifier},
        )

    if not _ax_passed:
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel(
            str(
                timeseries.columns[0]
                if isinstance(timeseries, pd.DataFrame)
                else timeseries.name
            )
        )
        ax.set_xlim(event_window)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        df_result = pd.DataFrame(
            {
                "time": new_index,
                "mean": binned_mean,
                "lower": binned_spread[0],
                "upper": binned_spread[1],
            }
        )
        _summary_data[_tup] = df_result
    return ax, _summary_data


def plot_session_trials(
    dataset: SessionDataset,
    *,
    alpha: float = 0.5,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    passed_ax = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 5)))

    trials = dataset.trials

    prob_ax = ax.twinx()
    yy_labels = []
    yy_ticks = []
    for patch_index in trials["patch_index"].unique():
        subset = trials[trials["patch_index"] == patch_index]

        ys_gain = 0.1
        yy_offset = -0.2 if patch_index == 0 else +0.2  # first patch gets an offset
        yy_ticks.append(patch_index + ys_gain / 2 + yy_offset)
        yy_labels.append(patch_index)

        ax.scatter(
            subset.index,
            subset["patch_index"] + subset["is_choice"] * ys_gain + yy_offset,
            color=patch_index_colormap[patch_index],
            label=patch_index,
            alpha=1,
            linewidths=subset["is_choice"] + 0.5,
            s=subset["is_choice"] * 20 + 50,
            marker="|",
        )

        rewarded_filter = (subset["is_choice"] == 1) & (subset["is_rewarded"] == 1)
        ax.scatter(
            subset[rewarded_filter].index,
            subset[rewarded_filter]["patch_index"]
            + subset[rewarded_filter]["is_choice"] * ys_gain * 2
            + yy_offset,
            color=patch_index_colormap[patch_index],
            label=patch_index,
            alpha=1,
            linewidths=1,
            s=subset[rewarded_filter]["is_choice"] * 20,
            marker="v",
        )

        mov_average = subset["is_choice"].ewm(alpha=alpha, adjust=False).mean()
        prob_ax.plot(
            subset.index,
            mov_average,
            color=patch_index_colormap[patch_index],
            label=f"PChoice(patch={patch_index})",
        )

        p_reward = subset["p_reward"]
        prob_ax.scatter(
            subset.index,
            p_reward,
            color=patch_index_colormap[patch_index],
            alpha=0.5,
            linestyle="--",
            label=f"PReward(patch={patch_index})",
        )

    if not passed_ax:
        ax.set_yticks(yy_ticks)
        ax.set_yticklabels((f"Patch {label}" for label in yy_labels))
        ax.set_xlabel("Trial")
        prob_ax.set_ylabel("Probability")
        prob_ax.set_yticks([0, 0.5, 1.0])
        prob_ax.set_ylim(-0.5, 1.5)
        prob_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        prob_ax.set_ylabel("Choice Probability")
    return ax


@contextmanager
def a_lot_of_style(
    font_scale=1.2,
    line_width=2,
    grid=True,
    despine=True,
    ticks_out=True,
):
    old_params = plt.rcParams.copy()

    plt.style.use("default")
    plt.rcParams.update(
        {
            # Fonts
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 11 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 9 * font_scale,
            "font.family": "DejaVu Sans",
            # Lines and markers
            "lines.linewidth": line_width,
            "lines.markersize": 6 * font_scale,
            # Axes and grid
            "axes.spines.top": not despine,
            "axes.spines.right": not despine,
            "axes.grid": grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            # Ticks
            "xtick.direction": "out" if ticks_out else "in",
            "ytick.direction": "out" if ticks_out else "in",
            "xtick.major.size": 4 * font_scale,
            "ytick.major.size": 4 * font_scale,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    try:
        yield
    finally:
        plt.rcParams.update(old_params)
