import dataclasses
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydantic_settings
import semver
from aind_behavior_vr_foraging import __semver__ as vrf_version
from aind_behavior_vr_foraging import task_logic as vrf_task
from pydantic import BaseModel, Field


class ProcessingSettings(BaseModel):
    downsample_position_to: Optional[float] = 60  # Hz, if None, do not downsample
    lickometer_refractory_period_s: float = 0.02  # seconds


class DataLoadingSettings(pydantic_settings.BaseSettings, yaml_file="sessions.yaml"):
    root_path: list[Path] = Field(description="Root path to the data directory")
    root_derived_path: Path = Field(Path("./derived"), description="Root path to the derived data directory")
    session_manifest_path: Path = Field(Path("./sessions.csv"), description="Path to the session manifest CSV file")
    dataset_version: str = Field(default=vrf_version, description="Version of the dataset to use")
    processing_settings: "ProcessingSettings" = Field(default=ProcessingSettings(), validate_default=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.PydanticBaseSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        """Specify order of settings sources (yaml file, env vars, etc)"""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            pydantic_settings.YamlConfigSettingsSource(settings_cls),
        )


@dataclasses.dataclass
class SessionInfo:
    subject: str
    session_id: str
    date: datetime.date
    data_path: Path
    version: semver.Version


@dataclasses.dataclass
class Trial:
    odor_onset_time: float
    choice_time: Optional[float]
    reward_time: Optional[float]
    reaction_duration: Optional[float]
    patch_index: int
    is_rewarded: Optional[bool]
    is_choice: bool
    p_reward: float
    stop_time: Optional[float]
    longest_stop_duration: Optional[float]


@dataclasses.dataclass
class ProcessedLickometer:
    onsets: np.ndarray
    frequency: pd.DataFrame


@dataclasses.dataclass
class ProcessedStreams:
    position_velocity: pd.DataFrame
    lickometer: ProcessedLickometer
    sniff_ipi_frequency: Optional[pd.DataFrame]


@dataclasses.dataclass
class SessionMetrics:
    total_distance: float  # total distance travelled
    reward_site_count: int  # number of reward sites observed
    stop_count: int  # number of stops/harvest attempts
    reward_count: int  # number of collected reward events
    session_duration: datetime.timedelta  # duration of the session
    p_stop_per_odor: dict[int, float]  # probability of stopping per odor
    total_reward_ml: float = 0.0  # total reward collected in mL


@dataclasses.dataclass
class Site:
    patch: dict
    site: dict
    patch_idx: int
    site_label: vrf_task.VirtualSiteLabels
    t_start: float
    t_end: float
