import typing as t
import pandas as pd
import numpy as np
import aind_behavior_vr_foraging.rig as vrf_rig
import aind_behavior_vr_foraging.task_logic as vrf_task
import logging
import contraqctor

from .models import Trial, Site
import dataclasses

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
    assert rig_settings.harp_treadmill.calibration is not None, (
        "Treadmill calibration is missing"
    )
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


def process_lickometer(
    dataset: contraqctor.contract.Dataset, *, refractory_period_s: float = 0.02
) -> np.ndarray:
    lickometer = (
        dataset.at("Behavior")
        .at("HarpLickometer")
        .load()
        .at("LickState")
        .load()
        .data.copy()
    )
    lickometer = lickometer[lickometer["MessageType"] == "EVENT"]["Channel0"]
    lick_onsets = lickometer[
        (lickometer) & (~lickometer.shift(1, fill_value=False))
    ].index
    return lick_onsets.values


def parse_trials(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    reward_label = vrf_task.VirtualSiteLabels.REWARDSITE
    rewarded_sites = (
        dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy()
    )
    rewarded_sites = rewarded_sites[
        rewarded_sites["data"].apply(lambda d: d["label"] == reward_label)
    ]

    # Merge nearest patch (backward in time)
    merged = pd.merge_asof(
        rewarded_sites,
        dataset.at("Behavior")
        .at("SoftwareEvents")
        .at("ActivePatch")
        .load()
        .data[["data"]],
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_patch"),
    )

    merged.rename(columns={"data_patch": "patches"}, inplace=True)
    merged["patch_index"] = merged.patches.apply(lambda d: d["state_index"])

    speaker_choice = (
        dataset.at("Behavior")
        .at("HarpBehavior")
        .load()
        .at("PwmStart")
        .load()
        .data.copy()
    )
    speaker_choice = speaker_choice[
        (speaker_choice["MessageType"] == "WRITE") & (speaker_choice["PwmDO2"])
    ]

    water_delivery = (
        dataset.at("Behavior")
        .at("HarpBehavior")
        .load()
        .at("OutputSet")
        .load()
        .data.copy()
    )
    water_delivery = water_delivery[
        (water_delivery["MessageType"] == "WRITE") & (water_delivery["SupplyPort0"])
    ]["SupplyPort0"]

    odor_onset = (
        dataset.at("Behavior")
        .at("HarpOlfactometer")
        .load()
        .at("EndValveState")
        .load()
        .data
    )
    odor_onset = odor_onset[odor_onset["MessageType"] == "WRITE"]["EndValve0"]
    odor_onset = odor_onset[(odor_onset) & (~odor_onset.shift(1, fill_value=False))]

    patches_state = (
        dataset.at("Behavior").at("SoftwareEvents").at("PatchState").load().data.copy()
    )
    expanded = pd.json_normalize(patches_state["data"])
    expanded.index = patches_state.index
    patches_state = patches_state.join(expanded)

    trials = []

    i = 0
    for i in range(len(merged) - 1):
        this_timestamp = merged.index[i]
        next_timestamp = merged.index[i + 1]
        logger.debug(f"Processing trial {i} at {this_timestamp} - {next_timestamp}")
        ## Find closest odor_onset after this_timestamp but before next_timestamp
        odor_onsets_in_interval = odor_onset[
            (odor_onset.index >= this_timestamp) & (odor_onset.index < next_timestamp)
        ]
        if len(odor_onsets_in_interval) == 0:
            logger.warning(
                f"No odor onset in site {i} interval...Using software event instead"
            )
            odor_onsets_in_interval = merged.iloc[this_timestamp]

        ## Find closest speaker_choice after this_timestamp but before next_timestamp
        speaker_choices_in_interval = speaker_choice[
            (speaker_choice.index >= this_timestamp)
            & (speaker_choice.index < next_timestamp)
        ]
        assert len(speaker_choices_in_interval) <= 1, (
            "Multiple speaker choices in interval"
        )
        water_deliveries_in_interval = water_delivery[
            (water_delivery.index >= this_timestamp)
            & (water_delivery.index < next_timestamp)
        ]
        if len(water_deliveries_in_interval) > 1:
            logger.warning(
                f"Multiple water deliveries in interval {this_timestamp} - {next_timestamp}"
            )
            water_deliveries_in_interval = water_deliveries_in_interval.iloc[:1]

        # Get the FIRST patch state AFTER the this_timestamp
        site_state_at_reward = patches_state[
            (patches_state.index > this_timestamp)
            & (patches_state["PatchId"] == merged.iloc[i]["patch_index"])
        ].iloc[0]

        trial = Trial(
            odor_onset_time=odor_onsets_in_interval.index[0],
            choice_time=speaker_choices_in_interval.index[0]
            if len(speaker_choices_in_interval) == 1
            else None,
            reward_time=water_deliveries_in_interval.index[0]
            if len(water_deliveries_in_interval) == 1
            else None,
            reaction_duration=(
                speaker_choices_in_interval.index[0] - odor_onsets_in_interval.index[0]
            )
            if len(speaker_choices_in_interval) == 1
            else None,
            patch_index=merged.iloc[i]["patch_index"],
            is_rewarded=len(water_deliveries_in_interval) == 1,
            p_reward=site_state_at_reward["Probability"],
            is_choice=len(speaker_choices_in_interval) == 1,
        )
        trials.append(trial)

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
    sites = (
        dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy()
    )
    patches = (
        dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data.copy()
    )

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
