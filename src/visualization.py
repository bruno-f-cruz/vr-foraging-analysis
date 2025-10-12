from matplotlib import pyplot as plt
import numpy as np
from aind_behavior_vr_foraging import task_logic as vrf_task
from typing import Optional
from .dataset import SessionDataset

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
        trials["choice_time"].index,
        np.ones_like(trials["choice_time"]) * 0.5,
        color="C3",
        label="Choices",
        marker="|",
        s=200,
        lw=2,
    )
    ax2.scatter(
        trials["reward_time"].index,
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

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    return ax, ax2
