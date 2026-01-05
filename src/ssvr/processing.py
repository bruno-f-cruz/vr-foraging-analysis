import dataclasses
import logging
import typing as t
from functools import partial
from typing import Callable

import aind_behavior_vr_foraging.rig as vrf_rig
import aind_behavior_vr_foraging.task_logic as vrf_task
import contraqctor
import numpy as np
import pandas as pd

from .models import ProcessedLickometer, Site, Trial

logger = logging.getLogger(__name__)


def compute_position_and_velocity(
    dataset: contraqctor.contract.Dataset, *, downsample_to_hz: t.Optional[float]
) -> pd.DataFrame:
    """Computes position and velocity from treadmill encoder data"""
    dataset.at("Behavior").at("InputSchemas").load_all()
    rig_settings = t.cast(
        vrf_rig.AindVrForagingRig,
        dataset.at("Behavior").at("InputSchemas").at("Rig").load().data,
    )

    ## Parse velocity and position
    treadmill_data = t.cast(
        pd.DataFrame,
        dataset.at("Behavior").at("HarpTreadmill").load().at("SensorData").load().data,
    )
    encoder = treadmill_data.query("MessageType == 'EVENT'")["Encoder"].copy()
    assert rig_settings.harp_treadmill.calibration is not None, "Treadmill calibration is missing"
    calibration = rig_settings.harp_treadmill.calibration
    converting_factor = (
        calibration.output.wheel_diameter
        * np.pi
        / calibration.output.pulses_per_revolution
        * (-1 if calibration.output.invert_direction else 1)
    )
    position = (encoder - encoder.iloc[0]) * converting_factor

    displacement = position.diff().fillna(0)
    velocity = displacement / position.index.to_series().diff().fillna(1)
    df = pd.DataFrame({"position": position, "velocity": velocity})
    if downsample_to_hz is None:
        return df
    if not np.issubdtype(df.index.dtype, np.number):
        df.index = pd.to_numeric(df.index, errors="coerce")
    df.sort_index(inplace=True)
    df.index = pd.to_timedelta(df.index, unit="s")

    # Compute interval as a Timedelta
    dt = pd.to_timedelta(1.0 / downsample_to_hz, unit="s")

    # Use timedelta directly
    df = df.resample(dt, label="right", closed="right").mean()
    df.dropna(inplace=True)
    df.index = df.index.total_seconds()
    return df


def process_sniff_detector(
    dataset: contraqctor.contract.Dataset,
    *,
    notch_filter_freq: t.Optional[float] = 60.0,
) -> t.Optional[pd.DataFrame]:
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt, find_peaks, iirnotch

    try:
        data = dataset.at("Behavior").at("HarpSniffDetector").at("RawVoltage").load().data
        data = data[data["MessageType"] == "EVENT"]
        fs = (
            dataset.at("Behavior")
            .at("HarpSniffDetector")
            .load()
            .at("RawVoltageDispatchRate")
            .load()
            .data.iloc[-1]
            .values[0]
        )
    except Exception as e:
        logging.warning(f"Failed to load SniffDetector data: {e}")
        return None

    t = data.index.values
    signal = data["RawVoltage"].values
    dt = 1.0 / fs
    t_uniform = np.arange(t[0], t[-1], dt)
    interp_func = interp1d(t, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    y_uniform = interp_func(t_uniform)
    if notch_filter_freq is not None:
        b_notch, a_notch = iirnotch(notch_filter_freq, 30.0, fs)
        y_uniform = filtfilt(b_notch, a_notch, y_uniform)

    b, a = butter(4, [0.2, 15], btype="bandpass", fs=fs)
    y_filtered = filtfilt(b, a, y_uniform)

    peaks, _ = find_peaks(y_filtered, height=0.5 * np.std(y_filtered), prominence=2.5)
    ipi = np.diff(t_uniform[peaks])
    frequency = 1.0 / ipi

    return pd.DataFrame(
        {
            "ipi": ipi,
            "frequency": frequency,
        },
        index=pd.Index(t_uniform[peaks][1:], name="Seconds"),
    )


def process_lickometer(
    dataset: contraqctor.contract.Dataset, *, refractory_period_s: float = 0.02, dt_resample: float = 0.1
) -> ProcessedLickometer:
    lickometer = dataset.at("Behavior").at("HarpLickometer").load().at("LickState").load().data.copy()
    lickometer = lickometer[lickometer["MessageType"] == "EVENT"]["Channel0"]
    lick_onsets = lickometer[(lickometer) & (~lickometer.shift(1, fill_value=False))].index
    if len(lick_onsets) == 0:
        return ProcessedLickometer(
            onsets=np.array([]),
            frequency=pd.DataFrame(
                {"frequency": []},
                index=pd.Index([], name="Seconds"),
            ),
        )

    keep = np.ones(len(lick_onsets), dtype=bool)
    keep[1:] = np.diff(lick_onsets) >= refractory_period_s
    kept = lick_onsets[keep]

    t_start = lickometer.index.values[0]
    t_end = lickometer.index.values[-1]

    bin_edges = np.arange(t_start, t_end + dt_resample, dt_resample)
    counts, _ = np.histogram(kept, bins=bin_edges)

    frequency_hz = counts / dt_resample

    bin_centers = bin_edges[:-1] + dt_resample / 2

    frequency = pd.Series(
        frequency_hz,
        index=pd.Index(bin_centers, name="Seconds"),
        name="frequency",
    )

    return ProcessedLickometer(
        onsets=kept,
        frequency=frequency,
    )


def parse_trials(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    reward_label = vrf_task.VirtualSiteLabels.REWARDSITE
    rewarded_sites = dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy()
    rewarded_sites = rewarded_sites[rewarded_sites["data"].apply(lambda d: d["label"] == reward_label)]

    # Merge nearest patch (backward in time)
    merged = pd.merge_asof(
        rewarded_sites,
        dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data[["data"]],
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_patch"),
    )

    merged.rename(columns={"data_patch": "patches"}, inplace=True)
    merged["patch_index"] = merged.patches.apply(lambda d: d["state_index"])

    speaker_choice = dataset.at("Behavior").at("HarpBehavior").load().at("PwmStart").load().data.copy()
    speaker_choice = speaker_choice[(speaker_choice["MessageType"] == "WRITE") & (speaker_choice["PwmDO2"])]

    water_delivery = dataset.at("Behavior").at("HarpBehavior").load().at("OutputSet").load().data.copy()
    water_delivery = water_delivery[(water_delivery["MessageType"] == "WRITE") & (water_delivery["SupplyPort0"])][
        "SupplyPort0"
    ]

    odor_onset = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
    odor_onset = odor_onset[odor_onset["MessageType"] == "WRITE"]["EndValve0"]
    odor_onset = odor_onset[(odor_onset) & (~odor_onset.shift(1, fill_value=False))]

    patches_state = dataset.at("Behavior").at("SoftwareEvents").at("PatchState").load().data.copy()
    expanded = pd.json_normalize(patches_state["data"])
    expanded.index = patches_state.index
    patches_state = patches_state.join(expanded)

    patches_state_at_reward = dataset.at("Behavior").at("SoftwareEvents").at("PatchStateAtReward").load().data.copy()
    expanded = pd.json_normalize(patches_state_at_reward["data"])
    expanded.index = patches_state_at_reward.index
    patches_state_at_reward = patches_state_at_reward.join(expanded)

    trials: list[Trial] = []

    i = 0
    for i in range(len(merged) - 1):
        this_timestamp = merged.index[i]
        next_timestamp = merged.index[i + 1]
        logger.debug(f"Processing trial {i} at {this_timestamp} - {next_timestamp}")
        ## Find closest odor_onset after this_timestamp but before next_timestamp
        odor_onsets_in_interval = odor_onset[(odor_onset.index >= this_timestamp) & (odor_onset.index < next_timestamp)]
        if len(odor_onsets_in_interval) == 0:
            logger.warning(f"No odor onset in site {i} interval...Using software event instead")
            odor_onsets_in_interval = merged.loc[[this_timestamp]]

        ## Find closest speaker_choice after this_timestamp but before next_timestamp
        speaker_choices_in_interval = speaker_choice[
            (speaker_choice.index >= this_timestamp) & (speaker_choice.index < next_timestamp)
        ]
        assert len(speaker_choices_in_interval) <= 1, "Multiple speaker choices in interval"

        stops = dataset.at("Behavior").at("OperationControl").at("IsStopped").data
        ## Find the closest is stop
        if len(speaker_choices_in_interval) > 0:
            mask = (stops.index <= speaker_choices_in_interval.index[0]) & (stops.iloc[:, 0])
            stops_before_speaker = stops.loc[mask]
            if len(stops_before_speaker) > 0:
                stop_time = stops_before_speaker.index[-1]  # Get the closest one before
            else:
                raise ValueError("No stop found before speaker choice")
        else:
            stop_time = None

        ## Find the longest stop inside the interval
        stop_data_inside_interval = stops[(stops.index >= this_timestamp) & (stops.index < next_timestamp)]
        if not stop_data_inside_interval.empty:
            # If the first value is True, we compute the duration from the odor onset
            if stop_data_inside_interval.iloc[0, 0]:
                prepend = pd.DataFrame(
                    [[False]], index=[odor_onsets_in_interval.index[0]], columns=stop_data_inside_interval.columns
                )
                stop_data_inside_interval = pd.concat([prepend, stop_data_inside_interval])

            index_diff = np.diff(stop_data_inside_interval.index.values)
            mask = stop_data_inside_interval.values.flatten()
            longest_stop_duration = index_diff[mask[:-1]].max() if len(index_diff[mask[:-1]]) > 0 else None
        else:
            longest_stop_duration = None

        ## Find closest water_delivery after this_timestamp but before next_timestamp
        water_deliveries_in_interval = water_delivery[
            (water_delivery.index >= this_timestamp) & (water_delivery.index < next_timestamp)
        ]
        if len(water_deliveries_in_interval) > 1:
            logger.warning(f"Multiple water deliveries in interval {this_timestamp} - {next_timestamp}")
            water_deliveries_in_interval = water_deliveries_in_interval.iloc[:1]

        # Get the FIRST patch state AFTER the this_timestamp
        site_state_at_reward = patches_state_at_reward[
            (patches_state_at_reward.index > this_timestamp)
            & (patches_state_at_reward["PatchId"] == merged.iloc[i]["patch_index"])
        ]
        if len(site_state_at_reward) > 0:
            site_state_at_reward = site_state_at_reward.iloc[0]
            # TODO this is because of block switches...
            trial = Trial(
                odor_onset_time=odor_onsets_in_interval.index[0],
                choice_time=speaker_choices_in_interval.index[0] if len(speaker_choices_in_interval) == 1 else None,
                reward_time=water_deliveries_in_interval.index[0] if len(water_deliveries_in_interval) == 1 else None,
                reaction_duration=(speaker_choices_in_interval.index[0] - odor_onsets_in_interval.index[0])
                if len(speaker_choices_in_interval) == 1
                else None,
                patch_index=merged.iloc[i]["patch_index"],
                is_rewarded=len(water_deliveries_in_interval) == 1,
                p_reward=site_state_at_reward["Probability"],
                is_choice=len(speaker_choices_in_interval) == 1,
                stop_time=stop_time,
                longest_stop_duration=longest_stop_duration,
            )
            trials.append(trial)
        else:
            trials.append(
                Trial(
                    odor_onset_time=odor_onsets_in_interval.index[0],
                    choice_time=None,
                    reward_time=None,
                    reaction_duration=None,
                    patch_index=merged.iloc[i]["patch_index"],
                    is_rewarded=None,
                    p_reward=np.nan,
                    is_choice=False,
                    stop_time=None,
                    longest_stop_duration=None,
                )
            )

    trials_df = pd.DataFrame([trial.__dict__ for trial in trials])
    return trials_df


def get_closest_from_timestamp(
    timestamps: np.ndarray,
    df: pd.DataFrame,
    *,
    search_mode: t.Literal["closest", "next", "previous"] = "closest",
) -> np.ndarray:
    """
    For each timestamp in `timestamps`, find the index in df.index that is:
      - 'closest': closest in value
      - 'next': the first index >= timestamp
      - 'previous': the last index <= timestamp

    Returns an array of indices from df.index.
    """
    df_index = df.index.values

    # Use numpy searchsorted for efficient lookup
    timestamps = np.asarray(timestamps)
    if search_mode == "closest":
        idx_left = np.searchsorted(df_index, timestamps, side="left")
        idx_right = np.clip(idx_left - 1, 0, len(df_index) - 1)
        idx_left = np.clip(idx_left, 0, len(df_index) - 1)
        left_diff = np.abs(df_index[idx_left] - timestamps)
        right_diff = np.abs(df_index[idx_right] - timestamps)
        use_left = left_diff <= right_diff
        idxs = np.where(use_left, idx_left, idx_right)
    elif search_mode == "next":
        idxs = np.searchsorted(df_index, timestamps, side="left")
        idxs = np.clip(idxs, 0, len(df_index) - 1)
    elif search_mode == "previous":
        idxs = np.searchsorted(df_index, timestamps, side="right") - 1
        idxs = np.clip(idxs, 0, len(df_index) - 1)
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")
    return df.index[idxs]


def process_sites(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    sites = dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy()
    patches = dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data.copy()

    # Ensure patches and sites are sorted by index
    sites = sites.sort_index()
    patches = patches.sort_index()

    # Use merge_asof to efficiently find the closest preceding patch for each site
    sites = pd.merge_asof(
        sites.sort_index(),
        patches[["data"]].rename(columns={"data": "patch_data"}).sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    sites["label"] = sites["data"].apply(lambda d: d["label"]).values
    sites["patch_idx"] = sites["patch_data"].apply(lambda d: d["state_index"]).values

    all_sites = []
    for i in range(len(sites)):
        all_sites.append(
            Site(
                patch=sites["patch_data"].iloc[i],
                site=sites["data"].iloc[i],
                patch_idx=sites["patch_idx"].iloc[i],
                site_label=sites["label"].iloc[i],
                t_start=sites.index[i],
                t_end=sites.index[i + 1] if i < len(sites) - 1 else sites.index[i],
            )
        )
    df = pd.DataFrame([dataclasses.asdict(site) for site in all_sites])
    df["plot_label"] = df.apply(
        lambda row: f"{row['site_label']}_{row['patch_idx']}"
        if row["site_label"] == vrf_task.VirtualSiteLabels.REWARDSITE
        else row["site_label"],
        axis=1,
    )
    return df


_extra_column_timestamp_name: str = "_aligned_timestamp_"
_extra_column_index_name: str = "_alignment_index_"


def aligned_to(
    timestamps: np.ndarray | pd.Series | pd.DataFrame,
    timeseries: pd.Series | pd.DataFrame,
    *,
    event_window: tuple[float, float] = (-1, 1),
) -> pd.DataFrame:
    # Normalize/sanitize timestamps input
    if isinstance(timestamps, np.ndarray):
        assert timestamps.ndim == 1, "Timestamps array must be one-dimensional"
        timestamps_array = timestamps
    elif isinstance(timestamps, (pd.Series, pd.DataFrame)):
        timestamps_array = timestamps.index.to_numpy()
    else:
        timestamps_array = np.array(timestamps)

    if isinstance(timeseries, pd.Series):
        timeseries = timeseries.to_frame()

    assert _extra_column_timestamp_name not in timeseries.columns, (
        f"Column {_extra_column_timestamp_name} already exists in timeseries"
    )
    assert _extra_column_index_name not in timeseries.columns, (
        f"Column {_extra_column_index_name} already exists in timeseries"
    )

    snippets = []
    for idx, ts in enumerate(timestamps_array):
        _win = np.array(event_window) + ts
        mask = (timeseries.index >= _win[0]) & (timeseries.index <= _win[1])
        snippet = timeseries.loc[mask].copy()
        snippet[_extra_column_timestamp_name] = snippet.index - ts
        snippet[_extra_column_index_name] = idx
        snippets.append(snippet)

    return pd.concat(snippets, ignore_index=False) if snippets else timeseries.iloc[0:0].copy()


suffix_chunk_session_idx = "_chunk_session_idx_"


def aligned_to_grouped_by(
    timestamp_df: pd.DataFrame | list[pd.DataFrame],
    timeseries: pd.Series | list[pd.Series],
    by: list[t.Any] | None = None,
    timestamp_column: str | None = None,
    *,
    event_window: tuple[float, float] = (-1, 1),
) -> t.Tuple[dict[t.Any, pd.DataFrame], list[t.Any]]:
    if isinstance(timestamp_df, list) != isinstance(timeseries, list):
        raise ValueError("timestamp_df and timeseries must both be lists or both be single DataFrames/Series.")

    timestamp_df = [timestamp_df] if not isinstance(timestamp_df, list) else timestamp_df
    timeseries = [timeseries] if not isinstance(timeseries, list) else timeseries

    if len(timestamp_df) != len(timeseries):
        raise ValueError("If timestamp_df and timeseries are lists, they must be of the same length.")

    by = by or []

    _summary_data: dict[t.Any, list[pd.DataFrame]] = {}

    for session_idx, (session_timestamp_df, session_timeseries) in enumerate(zip(timestamp_df, timeseries)):
        for _tup, df in session_timestamp_df.groupby(by):
            if timestamp_column is not None:
                timestamps = df[timestamp_column].to_numpy()
            else:
                timestamps = df.index.to_numpy()

            data = aligned_to(
                timestamps,
                session_timeseries,
                event_window=event_window,
            )
            data[suffix_chunk_session_idx] = session_idx
            _summary_data.setdefault(_tup, []).append(data)

    _new_summary_data: dict[t.Any, pd.DataFrame] = {}
    for key in _summary_data:
        _new_summary_data[key] = pd.concat(_summary_data[key], ignore_index=True)
    return _new_summary_data, by


def is_one_dimensional_array(arr: np.ndarray) -> bool:
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input is not a numpy ndarray.")
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and 1 in arr.shape:
        return True
    if arr.ndim > 2 and all(dim == 1 for dim in arr.shape[2:]):
        return True
    return False


def summarize_aligned_to(
    aligned: pd.DataFrame,
    *,
    time_bin_width: float = 0.025,
    agg_fnc: Callable[[np.ndarray], np.ndarray] = partial(np.nanmean, axis=1),
    agg_spread_fnc: Callable[[np.ndarray], t.Iterable[np.ndarray] | np.ndarray] = partial(
        np.nanpercentile, q=[2.5, 97.5], axis=1
    ),
    agg_spread_flattening_axis: int = 0,
    data_column_name: t.Optional[str] = None,
    new_index: t.Optional[pd.Index] = None,
) -> pd.DataFrame:
    if new_index is None:
        new_index = pd.Index(
            np.arange(
                aligned[_extra_column_timestamp_name].min(),
                aligned[_extra_column_timestamp_name].max(),
                time_bin_width,
            )
        )

    data_column_name = data_column_name or aligned.columns[0]

    n_chunks = aligned[_extra_column_index_name].max()
    binned_data = np.full((len(new_index), n_chunks + 1), np.nan)
    for i_chunk, chunk_df in aligned.groupby(_extra_column_index_name):
        binned_data[:, i_chunk] = np.interp(
            new_index, chunk_df[_extra_column_timestamp_name], chunk_df[data_column_name].values
        )
    summarized_df = pd.DataFrame(
        {
            "agg": agg_fnc(binned_data),
        },
        index=new_index,
    )
    agg_spread = agg_spread_fnc(binned_data)

    if isinstance(agg_spread, np.ndarray):
        if is_one_dimensional_array(agg_spread):
            agg_spread = [np.squeeze(agg_spread)]
        elif agg_spread.ndim == 2:
            agg_spread = [np.squeeze(agg_spread[i, :]) for i in range(agg_spread.shape[agg_spread_flattening_axis])]
        else:
            raise ValueError("agg_spread_fnc returned an ndarray with more than one dimension.")

    if not isinstance(agg_spread, t.Iterable):
        raise ValueError("agg_spread_fnc must return an iterable of arrays.")

    for i, spread in enumerate(agg_spread):
        summarized_df[f"agg_spread_{i}"] = np.squeeze(spread)
    return summarized_df


def summarize_grouped_by(
    grouped: dict[t.Any, pd.DataFrame],
    *,
    time_bin_width: float = 0.025,
    agg_fnc: Callable[[np.ndarray], np.ndarray] = partial(np.nanmean, axis=1),
    agg_spread_fnc: Callable[[np.ndarray], t.Iterable[np.ndarray] | np.ndarray] = partial(
        np.nanpercentile, q=[2.5, 97.5], axis=1
    ),
    data_column_name: t.Optional[str] = None,
) -> dict[t.Any, pd.DataFrame]:
    summarized_by_group: dict[t.Any, pd.DataFrame] = {}
    new_index = pd.Index(
        np.arange(
            min(df[_extra_column_timestamp_name].min() for df in grouped.values()),
            max(df[_extra_column_timestamp_name].max() for df in grouped.values()),
            time_bin_width,
        )
    )

    data_column_name = data_column_name or grouped[next(iter(grouped))].columns[0]

    for group, df in grouped.items():
        _new_df = summarize_aligned_to(
            df,
            time_bin_width=time_bin_width,
            agg_fnc=agg_fnc,
            agg_spread_fnc=agg_spread_fnc,
            data_column_name=data_column_name,
            new_index=new_index,
        )
        summarized_by_group[group] = _new_df

    return summarized_by_group
