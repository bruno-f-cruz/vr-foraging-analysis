import typing as t
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_switches_df(
    all_trials_df: pd.DataFrame, block_switch_filter: t.Literal["same", "different", "both"] = "same"
) -> pd.DataFrame:
    """Get a DataFrame of block switches with the block probabilities before and after the switch."""

    block_switches = (
        all_trials_df["block_index"].diff().fillna(0) > 0
    )  # this also gets rid of cross session switches since those diffs will be <=0
    switch_indices = all_trials_df.index[block_switches]
    all_trials_df.sort_index(inplace=True)  # just to ensure the shift works correctly
    block_probabilities_before = all_trials_df["block_patch_probabilities"].shift()[block_switches]
    block_probabilities_after = all_trials_df["block_patch_probabilities"][block_switches]
    prob_switch_df = pd.DataFrame(
        {
            "before": block_probabilities_before.values,
            "after": block_probabilities_after.values,
            "before_high_index": [np.argmax(probs) for probs in block_probabilities_before.values],
            "after_high_index": [np.argmax(probs) for probs in block_probabilities_after.values],
            "after_low_index": [np.argmin(probs) for probs in block_probabilities_after.values],
            "before_low_index": [np.argmin(probs) for probs in block_probabilities_before.values],
        },
        index=switch_indices,
    )

    if block_switch_filter == "same":
        prob_switch_df = prob_switch_df[prob_switch_df["before"].apply(tuple) == prob_switch_df["after"].apply(tuple)]
    elif block_switch_filter == "different":
        prob_switch_df = prob_switch_df[prob_switch_df["before"].apply(tuple) != prob_switch_df["after"].apply(tuple)]
    elif block_switch_filter == "both":
        pass
    else:
        raise ValueError(f"Invalid block_switch_filter: {block_switch_filter}")

    return prob_switch_df


def calculate_choice_matrix(
    all_trials_df: pd.DataFrame,
    *,
    trial_window: t.Tuple[int, int] = (-10, 30),
    block_switch_filter: t.Literal["same", "different", "both"] = "same",
    column_name: str = "is_choice",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Calculate choice matrices around block switches for plotting. Returns a 3D numpy array of shape (num_switches, num_trials_in_window, 2) and a DataFrame of switch trials with original indices."""
    is_patch_low_after_zip = (0, 1)  # high, low
    prob_switch_df = get_switches_df(all_trials_df, block_switch_filter=block_switch_filter)

    switch_choice_data = np.full(
        shape=(len(prob_switch_df), trial_window[1] - trial_window[0], len(is_patch_low_after_zip)),
        fill_value=np.nan,
        dtype=float,
    )
    switch_indices = []
    for i_switch, (trial_switch, row) in tqdm(
        enumerate(prob_switch_df.iterrows()), desc="Processing block switches", total=len(prob_switch_df)
    ):
        switch_trial = all_trials_df.loc[trial_switch]
        switch_indices.append(trial_switch)
        session_id = switch_trial["session_id"]
        session_trials = all_trials_df[all_trials_df["session_id"] == session_id]
        trial_window_mask_after = (session_trials["trials_from_last_block_by_trial_type"] < trial_window[1]) & (
            session_trials["block_index"] == switch_trial["block_index"]
        )
        trial_window_mask_before = (session_trials["trials_to_next_block_by_trial_type"] < -trial_window[0]) & (
            session_trials["block_index"] == switch_trial["block_index"] - 1
        )
        trial_window_mask = trial_window_mask_after | trial_window_mask_before
        for is_patch_low_after in is_patch_low_after_zip:
            if is_patch_low_after == 0:
                patch_idx = prob_switch_df.loc[trial_switch]["after_high_index"]
            else:
                patch_idx = prob_switch_df.loc[trial_switch]["after_low_index"]
            patch_id_mask = session_trials["patch_index"] == patch_idx
            trials_to_take = session_trials[trial_window_mask & patch_id_mask]
            if len(trials_to_take) == 0:
                continue
            min_idx = trials_to_take.iloc[0]["trials_to_next_block_by_trial_type"]
            slice_from_array_start = -(trial_window[0] + min_idx) - 1
            slice_from_array_end = slice_from_array_start + len(trials_to_take)
            switch_choice_data[i_switch, slice_from_array_start:slice_from_array_end, is_patch_low_after] = (
                trials_to_take[column_name].values
            )

    switch_trials_df = all_trials_df.loc[switch_indices]
    switch_trials_df["index_ord"] = np.arange(len(switch_trials_df))
    switch_trials_df = switch_trials_df.join(prob_switch_df)

    return switch_choice_data, switch_trials_df


def _find_consecutive_run(sequence: np.ndarray, target_value: bool, n_consecutive: int) -> float:
    """
    Find the trial number where the first occurrence of n_consecutive target_value occurs.

    Parameters
    ----------
    sequence : np.ndarray
        Boolean array of choices
    target_value : bool
        Value to look for (True or False)
    n_consecutive : int
        Number of consecutive occurrences needed

    Returns
    -------
    float
        Trial number (1-indexed) where the run completes, or NaN if not found
    """
    if len(sequence) < n_consecutive:
        return np.nan

    consecutive_count = 0

    for i, value in enumerate(sequence):
        if value == target_value:
            consecutive_count += 1
            if consecutive_count == n_consecutive:
                return i + 1
        else:
            consecutive_count = 0

    return np.nan


def calculate_consecutive_choice_runs(
    all_trials_df: pd.DataFrame, switch_trials_df: pd.DataFrame, *, n_consecutive: int = 3, max_trials_ahead: int = 50
) -> pd.DataFrame:
    """
    Calculate the number of trials after each switch to achieve N consecutive choices.

    For each switch and each patch type, calculates how many trials it takes after
    the switch to have N consecutive is_choice == True and N consecutive is_choice == False.

    Parameters
    ----------
    all_trials_df : pd.DataFrame
        DataFrame containing all trials data
    switch_trials_df : pd.DataFrame
        DataFrame containing switch trial information with original indices
    n_consecutive : int, optional
        Number of consecutive choices to look for, by default 3
    max_trials_ahead : int, optional
        Maximum number of trials to look ahead after switch, by default 50

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: trial_index, patch_id, trials_to_n_consecutive_true,
        trials_to_n_consecutive_false
    """
    results = []

    for switch_idx, switch_trial in tqdm(
        switch_trials_df.iterrows(), desc="Calculating consecutive runs", total=len(switch_trials_df)
    ):
        session_id = all_trials_df.loc[switch_idx]["session_id"]
        session_trials = all_trials_df[all_trials_df["session_id"] == session_id]

        # Get patch indices for this switch
        high_patch_id = switch_trial["after_high_index"]
        low_patch_id = switch_trial["after_low_index"]

        for patch_id in [high_patch_id, low_patch_id]:
            # Get trials after switch for this patch
            after_switch_mask = (
                (session_trials["trials_from_last_block_by_trial_type"] >= 0)
                & (session_trials["trials_from_last_block_by_trial_type"] <= max_trials_ahead)
                & (session_trials["block_index"] == all_trials_df.loc[switch_idx]["block_index"])
                & (session_trials["patch_index"] == patch_id)
            )

            after_switch_trials = session_trials[after_switch_mask].sort_values("trials_from_last_block_by_trial_type")

            if len(after_switch_trials) == 0:
                results.append(
                    {
                        "trial_index": switch_idx,
                        "patch_id": patch_id,
                        "trials_to_n_consecutive_true": np.nan,
                        "trials_to_n_consecutive_false": np.nan,
                    }
                )
                continue

            choice_sequence = after_switch_trials["is_choice"].values
            trials_to_true = _find_consecutive_run(choice_sequence, target_value=True, n_consecutive=n_consecutive)
            trials_to_false = _find_consecutive_run(choice_sequence, target_value=False, n_consecutive=n_consecutive)

            results.append(
                {
                    "trial_index": switch_idx,
                    "patch_id": patch_id,
                    "trials_to_n_consecutive_true": trials_to_true,
                    "trials_to_n_consecutive_false": trials_to_false,
                }
            )

    results_df = pd.DataFrame(results)

    # Merge with all_trials_df and switch_trials_df using trial_index
    all_trials_reset = all_trials_df.reset_index()
    loser_cols = all_trials_reset.columns.intersection(results_df.columns).difference(["index"])
    results_df = results_df.merge(
        all_trials_reset.drop(columns=loser_cols), left_on="trial_index", right_on="index", how="left"
    )

    switch_trials_reset = switch_trials_df.reset_index()
    loser_cols = switch_trials_reset.columns.intersection(results_df.columns).difference(["index"])
    results_df = results_df.merge(
        switch_trials_reset.drop(columns=loser_cols), left_on="trial_index", right_on="index", how="left"
    )

    results_df["is_low_reward_patch"] = results_df["patch_id"] == results_df["after_low_index"]

    return results_df


def plot_block_switch_choice_patterns(
    switch_choice_data: np.ndarray,
    trial_window: t.Tuple[int, int],
    *,
    figsize: t.Tuple[float, float] = (12, 15),
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> t.Tuple[plt.Figure, t.List[plt.Axes]]:
    """
    Plot choice patterns around block switches.

    Parameters
    ----------
    switch_choice_data : np.ndarray
        3D array of shape (n_switches, n_trials_in_window, 2) containing choice data
        for each switch, where last dimension is [high_reward_patch, low_reward_patch]
    trial_window : tuple[int, int]
        Window around block switch (trials before, trials after)
    figsize : tuple[float, float]
        Figure size, by default (12, 15)
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure, by default None
    **kwargs
        Additional keyword arguments for customization

    Returns
    -------
    tuple[plt.Figure, list[plt.Axes]]
        Figure and list of axes objects [heatmap1, heatmap2, average_plot]
    """
    x_positions = np.arange(trial_window[0], trial_window[1])

    patch_names = ["High Reward Patch", "Low Reward Patch"]
    patch_colors = ["red", "blue"]

    _ax_passed = ax is not None
    if ax is None:
        # Create layout: 2x2 grid with bottom plot spanning full width
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        axes = [
            fig.add_subplot(gs[0, 0]),  # Top left - High reward patch heatmap
            fig.add_subplot(gs[0, 1]),  # Top right - Low reward patch heatmap
            fig.add_subplot(gs[1, :]),  # Bottom - Average choice probability (spans both columns)
        ]
    else:
        fig = ax.figure
        axes = [ax]

    if not _ax_passed:
        for patch_id in range(2):
            im = axes[patch_id].imshow(
                switch_choice_data[:, :, patch_id], aspect="auto", cmap="RdYlBu_r", interpolation="none", vmin=0, vmax=1
            )
            axes[patch_id].set_title(f"{patch_names[patch_id]} - Individual Switches")
            axes[patch_id].set_ylabel("Block Switch Number")

            tick_positions = np.arange(0, len(x_positions), 10)
            tick_labels = x_positions[tick_positions]
            axes[patch_id].set_xticks(tick_positions)
            axes[patch_id].set_xticklabels(tick_labels)
            axes[patch_id].set_xlabel("Trials Relative to Block Switch")

            switch_position = -trial_window[0]
            axes[patch_id].axvline(x=switch_position, color="white", linestyle="--", alpha=0.8)

            plt.colorbar(im, ax=axes[patch_id], label="Choice Probability")

        for patch_id in range(2):
            data = switch_choice_data[:, :, patch_id]
            mean_choice = np.nanmean(data, axis=0)

            # Bootstrap confidence intervals
            n_bootstrap = 1000
            confidence_level = 0.95
            alpha = 1 - confidence_level

            ci_lower = np.full_like(mean_choice, np.nan)
            ci_upper = np.full_like(mean_choice, np.nan)

            for i in range(len(mean_choice)):
                # Get valid (non-NaN) samples for this time point
                valid_samples = data[:, i][~np.isnan(data[:, i])]

                if len(valid_samples) > 1:  # Need at least 2 samples for bootstrap
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        resampled = np.random.choice(valid_samples, size=len(valid_samples), replace=True)
                        bootstrap_means.append(np.mean(resampled))

                    ci_lower[i] = np.percentile(bootstrap_means, 100 * alpha / 2)
                    ci_upper[i] = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

            axes[2].plot(
                x_positions,
                mean_choice,
                "o-",
                color=patch_colors[patch_id],
                alpha=0.8,
                label=patch_names[patch_id],
                linewidth=2,
            )
            axes[2].fill_between(x_positions, ci_lower, ci_upper, alpha=0.2, color=patch_colors[patch_id])

        axes[2].set_title("Average Choice Probability - Both Patches")
        axes[2].set_xlabel("Trials Relative to Block Switch")
        axes[2].set_ylabel("Choice Probability")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(x=0, color="black", linestyle="--", alpha=0.8, label="Block Switch")
        axes[2].legend()

    return fig, axes


def plot_trials_to_criterion_histogram(
    consecutive_runs_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: t.Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    plot_kernel: bool = True,
) -> t.Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of trials to reach criterion for high and low reward patches.

    Parameters
    ----------
    consecutive_runs_df : pd.DataFrame
        DataFrame containing consecutive runs data
    ax : plt.Axes, optional
        Axes to plot on, by default None
    figsize : tuple[float, float], optional
        Figure size if creating new figure, by default (8, 6)
    title : str, optional
        Plot title, by default None
    plot_kernel : bool, optional
        Whether to overlay kernel density estimate, by default True

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    low_reward_data = consecutive_runs_df[consecutive_runs_df["is_low_reward_patch"]]["trials_to_n_consecutive_false"]
    low_reward_data = low_reward_data.dropna()
    ax.hist(low_reward_data, bins=range(0, 51), alpha=0.5, label="Low reward patch", color="b", density=True)

    high_reward_data = consecutive_runs_df[~consecutive_runs_df["is_low_reward_patch"]]["trials_to_n_consecutive_true"]
    high_reward_data = high_reward_data.dropna()
    ax.hist(high_reward_data, bins=range(0, 51), alpha=0.5, label="High reward patch", color="r", density=True)

    # Plot kernel density estimates
    if plot_kernel:
        from scipy.stats import gaussian_kde

        if len(low_reward_data) > 1:
            kde_low = gaussian_kde(low_reward_data)
            x_range = np.linspace(low_reward_data.min(), low_reward_data.max(), 200)
            ax.plot(x_range, kde_low(x_range), color="b", linewidth=2, alpha=0.8)

        if len(high_reward_data) > 1:
            kde_high = gaussian_kde(high_reward_data)
            x_range = np.linspace(high_reward_data.min(), high_reward_data.max(), 200)
            ax.plot(x_range, kde_high(x_range), color="r", linewidth=2, alpha=0.8)

    if len(low_reward_data) > 0:
        low_median = np.median(low_reward_data)
        ax.axvline(low_median, color="b", linestyle="-", linewidth=2, alpha=0.8, label=f"Low median: {low_median:.1f}")

    if len(high_reward_data) > 0:
        high_median = np.median(high_reward_data)
        ax.axvline(
            high_median, color="r", linestyle="-", linewidth=2, alpha=0.8, label=f"High median: {high_median:.1f}"
        )

    ax.set_xlabel("Trials to reach criterion")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    ax.legend()

    return fig, ax
