from .models import DataLoadingSettings, SessionInfo, FilterOn
import logging
import typing as t
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)


def load_sessions(
    settings: DataLoadingSettings,
) -> t.Generator[SessionInfo, None, None]:
    for subject, filter_on in settings.subject_filters.items():
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
