import dataclasses
import datetime
import json
import logging
from pathlib import Path

import contraqctor
import pandas as pd
import semver
from aind_behavior_vr_foraging import __semver__ as vrf_version
from aind_behavior_vr_foraging.data_contract import dataset

from .models import DataLoadingSettings, FilterOn, ProcessedStreams, ProcessingSettings, SessionInfo, SessionMetrics
from .processing import (
    compute_position_and_velocity,
    parse_trials,
    process_lickometer,
    process_sites,
    process_sniff_detector,
)

logger = logging.getLogger(__name__)


def find_session_info(
    settings: DataLoadingSettings,
) -> list[SessionInfo]:
    unique_sessions: list[SessionInfo] = []
    for root_path in settings.root_path:
        if not root_path.exists():
            logger.warning(f"Root path {root_path} does not exist. Skipping.")
            continue
        all_sessions = list(map(create_session_info, root_path.iterdir()))
        for session in all_sessions:
            if session.session_id in [s.session_id for s in unique_sessions]:
                continue

            if (len(settings.filters) == 0) or (
                any(is_accept_session(session, filter_on) for filter_on in settings.filters)
            ):
                unique_sessions.append(session)
    return unique_sessions


def create_session_info(session_path: Path) -> SessionInfo:
    """Uses session names in the form of:
    - 808728_2025-10-09T153828Z
    - scicomp's weird format: behavior_789917_2025-10-20_20-34-17 (read as UTC)
    """

    parts = session_path.stem.split("_")
    if parts[0] == "behavior":
        subject = parts[1]
        date_str = f"{parts[2]}T{parts[3].replace('-', ':')}"
        # Parse as Seattle timezone and convert to date
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        date = dt.date()
    else:
        # Handle standard format: 808728_2025-10-09T153828Z
        subject = parts[0]
        date_str = parts[1]
        date = datetime.datetime.fromisoformat(date_str).date()

    task_logic_schema = session_path / "Behavior" / "Logs" / "tasklogic_input.json"
    _json = json.loads(task_logic_schema.read_text())
    version = semver.Version.parse(_json["version"])

    return SessionInfo(
        subject=subject,
        session_id=session_path.stem,
        date=date,
        data_path=session_path,
        version=version,
    )


def is_accept_session(session: SessionInfo, filter: FilterOn):
    if (filter.start_date) and (session.date < filter.start_date):
        logger.debug(f"Session {session.session_id} on {session.date} rejected by start_date {filter.start_date}")
        return False
    if (filter.end_date) and (session.date > filter.end_date):
        logger.debug(f"Session {session.session_id} on {session.date} rejected by end_date {filter.end_date}")
        return False
    if (filter.session_ids) and (session.session_id not in filter.session_ids):
        logger.debug(f"Session {session.session_id} on {session.date} rejected by session_ids {filter.session_ids}")
        return False
    logger.debug(f"Session {session.session_id} on {session.date} accepted")
    if (filter.subjects) and (session.subject not in filter.subjects):
        logger.debug(
            f"Session {session.session_id} for subject {session.subject} rejected by subjects {filter.subjects}"
        )
        return False
    return True


@dataclasses.dataclass
class SessionDataset:
    session_info: SessionInfo
    processing_settings: ProcessingSettings = dataclasses.field(default_factory=ProcessingSettings)
    dataset_version: str = vrf_version
    dataset: contraqctor.contract.Dataset = dataclasses.field(init=False)
    processed_streams: "ProcessedStreams" = dataclasses.field(init=False)
    session_metrics: "SessionMetrics" = dataclasses.field(init=False)
    trials: pd.DataFrame = dataclasses.field(init=False)
    sites: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        self.dataset = dataset(self.session_info.data_path, self.dataset_version)
        self.add_processed_streams()
        self.add_sites()
        self.add_trials_and_metrics()

    def add_processed_streams(self):
        self.processed_streams = get_processed_data_streams(self.dataset, self.processing_settings)

    def add_sites(self):
        self.sites = process_sites(self.dataset)

    def add_trials_and_metrics(self):
        self.trials = parse_trials(self.dataset)

        self.session_metrics = SessionMetrics(
            total_distance=self.processed_streams.position_velocity["velocity"].abs().sum(),
            reward_site_count=len(self.trials),
            stop_count=self.trials["choice_time"].notna().sum(),
            reward_count=self.trials["reward_time"].notna().sum(),
            p_stop_per_odor={
                int(patch_id): df["reward_time"].notna().sum() / len(df)
                for patch_id, df in self.trials.groupby("patch_index")
            },
            session_duration=self._get_session_duration(self.dataset),
            total_reward_ml=self.dataset["Behavior"]["SoftwareEvents"]["GiveReward"].read()["data"].sum() * 1e-3,
        )

    @staticmethod
    def _get_session_duration(dataset: contraqctor.contract.Dataset) -> datetime.timedelta:
        clk_timestamps = dataset["Behavior"]["HarpClockGenerator"]["Counter"].data
        delta = clk_timestamps.index[-1] - clk_timestamps.index[0]
        return datetime.timedelta(seconds=delta)


def get_processed_data_streams(
    dataset: contraqctor.contract.Dataset, settings: ProcessingSettings
) -> "ProcessedStreams":
    return ProcessedStreams(
        position_velocity=compute_position_and_velocity(dataset, downsample_to_hz=settings.downsample_position_to),
        lickometer=process_lickometer(dataset, refractory_period_s=settings.lickometer_refractory_period_s),
        sniff_ipi_frequency=process_sniff_detector(dataset),
    )
