from .models import DataLoadingSettings, ProcessedStreams, SessionInfo, FilterOn
import logging
from pathlib import Path
import datetime
from .models import ProcessingSettings, SessionMetrics
from .processing import (
    compute_position_and_velocity,
    parse_trials,
    process_lickometer,
    process_sites,
)

from aind_behavior_vr_foraging import __semver__ as vrf_version
from aind_behavior_vr_foraging.data_contract import dataset
import contraqctor
import dataclasses
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def find_session_info(
    settings: DataLoadingSettings,
) -> list[SessionInfo]:
    unique_sessions: list[SessionInfo] = []
    for root_path in settings.root_path:
        for subject, filter_on in settings.subject_filters.items():
            subject_path = root_path / subject
            if not subject_path.exists():
                logger.debug(f"Subject path does not exist: {subject_path}")
                continue
            logger.debug(f"Subject: {subject}, Filter: {filter_on}")
            available_sessions = map(create_session_info, subject_path.iterdir())
            filtered_sessions = (
                session
                for session in available_sessions
                if is_accept_session(session, filter_on)
                and session.session_id not in [s.session_id for s in unique_sessions]
            )
            unique_sessions.extend(filtered_sessions)
    return unique_sessions


def create_session_info(session_path: Path) -> SessionInfo:
    """Uses session names in the form of:
    808728_2025-10-09T153828Z
    or scicomp's weird format: behavior_789917_2025-10-20_20-34-17"""

    parts = session_path.stem.split("_")
    if parts[0] == "behavior":
        subject = parts[1]
        date_str = f"{parts[2]}T{parts[3].replace('-', ':')}"
        # Parse as Seattle timezone and convert to date
        seattle_tz = pytz.timezone("America/Los_Angeles")
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        dt = seattle_tz.localize(dt)
        date = dt.date()
    else:
        # Handle standard format: 808728_2025-10-09T153828Z
        subject = parts[0]
        date_str = parts[1]
        date = datetime.datetime.fromisoformat(date_str).date()

    return SessionInfo(
        subject=subject,
        session_id=session_path.stem,
        date=date,
        data_path=session_path,
    )


def is_accept_session(session: SessionInfo, filter: FilterOn):
    if (filter.start_date) and (session.date < filter.start_date):
        logger.debug(
            f"Session {session.session_id} on {session.date} rejected by start_date {filter.start_date}"
        )
        return False
    if (filter.end_date) and (session.date > filter.end_date):
        logger.debug(
            f"Session {session.session_id} on {session.date} rejected by end_date {filter.end_date}"
        )
        return False
    if (filter.session_ids) and (session.session_id not in filter.session_ids):
        logger.debug(
            f"Session {session.session_id} on {session.date} rejected by session_ids {filter.session_ids}"
        )
        return False
    logger.debug(f"Session {session.session_id} on {session.date} accepted")
    return True


@dataclasses.dataclass
class SessionDataset:
    session_info: SessionInfo
    dataset_version: str = vrf_version
    dataset: contraqctor.contract.Dataset = dataclasses.field(init=False)
    processed_streams: "ProcessedStreams" = dataclasses.field(init=False)
    session_metrics: "SessionMetrics" = dataclasses.field(init=False)
    trials: pd.DataFrame = dataclasses.field(init=False)
    sites: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        self.dataset = dataset(self.session_info.data_path, self.dataset_version)

    def add_processed_streams(self, settings: ProcessingSettings):
        self.processed_streams = get_processed_data_streams(self.dataset, settings)

    def add_sites(self):
        self.sites = process_sites(self.dataset)

    def add_trials_and_metrics(self):
        self.trials = parse_trials(self.dataset)
        self.session_metrics = SessionMetrics(
            total_distance=self.processed_streams.position_velocity["velocity"]
            .abs()
            .sum(),
            reward_site_count=len(self.trials),
            stop_count=self.trials["choice_time"].notna().sum(),
            reward_count=self.trials["reward_time"].notna().sum(),
            p_stop_per_odor={
                int(patch_id): df["reward_time"].notna().sum() / len(df)
                for patch_id, df in self.trials.groupby("patch_index")
            },
        )


def get_processed_data_streams(
    dataset: contraqctor.contract.Dataset, settings: ProcessingSettings
) -> "ProcessedStreams":
    return ProcessedStreams(
        position_velocity=compute_position_and_velocity(
            dataset, downsample_to_hz=settings.downsample_position_to
        ),
        lick_onsets=process_lickometer(
            dataset, refractory_period_s=settings.lickometer_refractory_period_s
        ),
    )


def make_session_dataset(
    session_info: SessionInfo,
    processing_settings: ProcessingSettings,
) -> SessionDataset:
    session_dataset = SessionDataset(session_info)
    session_dataset.add_processed_streams(processing_settings)
    session_dataset.add_trials_and_metrics()
    session_dataset.add_sites()
    return session_dataset
