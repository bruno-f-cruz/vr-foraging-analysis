import logging

import requests
import semver

logger = logging.getLogger(__name__)


def get_version_from_sha(repo: str, commit_hash: str) -> semver.Version:
    if repo.endswith(".git"):
        repo = repo[:-4]
    if repo.startswith("https://github.com/"):
        repo = repo[len("https://github.com/") :]
    base_url = f"https://raw.githubusercontent.com/{repo}/{commit_hash}/"
    pyproject_url = base_url + "pyproject.toml"

    try:
        resp = requests.get(pyproject_url)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.strip().startswith("version"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    return semver.Version.parse(version)
    except Exception:
        logger.warning(
            "Could not resolve version from pyproject.toml for commit hash %s. Falling back to __init__.py",
            commit_hash,
        )

    try:
        setup_url = base_url + "src/aind_behavior_vr_foraging/__init__.py"
        resp = requests.get(setup_url)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.strip().startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    return semver.Version.parse(version)

    except Exception:
        logger.error(
            "Could not resolve version from __init__.py for commit hash %s.",
            commit_hash,
        )

    raise ValueError(
        "Could not find version information for commit hash %s in repo %s.",
        commit_hash,
        repo,
    )
