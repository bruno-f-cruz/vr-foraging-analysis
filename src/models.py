import pydantic_settings
from pydantic import Field, BaseModel
from pathlib import Path
import datetime
from typing import Optional
from aind_behavior_vr_foraging import __semver__ as vrf_version
from aind_behavior_vr_foraging import task_logic as vrf_task
import dataclasses
import pandas as pd
import numpy as np


class ProcessingSettings(BaseModel):
    downsample_position_to: Optional[float] = 60  # Hz, if None, do not downsample
    lickometer_refractory_period_s: float = 0.02  # seconds


class DataLoadingSettings(pydantic_settings.BaseSettings, yaml_file="sessions.yaml"):
    root_path: Path = Field(..., description="Root path to the data directory")
    subject_filters: dict[str, "FilterOn"] = Field(
        default_factory=dict, description="Dictionary of subject filters"
    )
    dataset_version: str = Field(
        default=vrf_version, description="Version of the dataset to use"
    )
    processing_settings: "ProcessingSettings" = Field(
        default=ProcessingSettings(), validate_default=True
    )

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


class FilterOn(BaseModel):
    start_date: Optional[datetime.date] = Field(
        default=None,
        description="Start date to filter session. If None, no filtering on start date.",
    )
    end_date: Optional[datetime.date] = Field(
        default=None,
        description="End date to filter session. If None, no filtering on end date.",
    )
    session_ids: Optional[list[str]] = Field(
        default=None,
        description="List of session IDs to filter. If None, no filtering on session IDs.",
    )


class SessionInfo(BaseModel):
    subject: str = Field(..., description="Subject identifier")
    session_id: str = Field(..., description="Session identifier")
    date: datetime.date = Field(..., description="Date of the session")
    data_path: Path = Field(..., description="Path to the session data file")


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


@dataclasses.dataclass
class ProcessedStreams:
    position_velocity: pd.DataFrame
    lick_onsets: np.ndarray


@dataclasses.dataclass
class SessionMetrics:
    total_distance: float  # total distance travelled
    reward_site_count: int  # number of reward sites observed
    stop_count: int  # number of stops/harvest attempts
    reward_count: int  # number of collected reward events
    p_stop_per_odor: dict[int, float]  # probability of stopping per odor


@dataclasses.dataclass
class Site:
    patch: dict
    site: dict
    patch_idx: int
    site_label: vrf_task.VirtualSiteLabels
    t_start: float
    t_end: float
