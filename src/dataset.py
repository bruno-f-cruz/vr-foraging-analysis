from .models import DataLoadingSettings, ProcessedStreams, SessionInfo, FilterOn
import logging
import typing as t
from pathlib import Path
import datetime
from .models import ProcessingSettings, SessionMetrics
from .processing import compute_position_and_velocity, parse_trials, process_lickometer

from aind_behavior_vr_foraging import __semver__ as vrf_version
from aind_behavior_vr_foraging.data_contract import dataset
import contraqctor
import dataclasses
import pandas as pd

logger = logging.getLogger(__name__)


def find_session_info(
    settings: DataLoadingSettings,
) -> t.Generator[SessionInfo, None, None]:
    for subject, filter_on in settings.subject_filters.items():
        if not (settings.root_path / subject).exists():
            logger.warning(
                f"Subject path does not exist: {settings.root_path / subject}"
            )
            continue
        logger.debug(f"Subject: {subject}, Filter: {filter_on}")
        available_sessions = map(
            create_session_info, (settings.root_path / subject).iterdir()
        )
        filtered_sessions = (
            session
            for session in available_sessions
            if is_accept_session(session, filter_on)
        )
        yield from filtered_sessions


def create_session_info(session_path: Path) -> SessionInfo:
    """Uses session names in the form of: 808728_2025-10-09T153828Z"""
    subject, date = session_path.stem.split("_")
    return SessionInfo(
        subject=subject,
        session_id=session_path.stem,
        date=datetime.datetime.fromisoformat(date).date(),
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

    def __post_init__(self):
        self.dataset = dataset(self.session_info.data_path, self.dataset_version)

    def add_processed_streams(self, settings: ProcessingSettings):
        self.processed_streams = get_processed_data_streams(self.dataset, settings)

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
    return session_dataset
