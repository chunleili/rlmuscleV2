"""Centralized logging setup for VMuscle.

Usage in examples::

    from muscle.log import setup_logging
    setup_logging()  # call once at entry point

Usage in library modules::

    import logging
    logging.getLogger(__name__).info("...")
"""

import logging
from pathlib import Path


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.DEBUG,
):
    """Configure the 'VMuscle' logger hierarchy and root-level console output.

    Args:
        log_file: Path to log file. Defaults to ``<project_root>/log.md``.
        level: Logging level for the VMuscle logger tree.
    """
    if log_file is None:
        log_file = Path(__file__).resolve().parents[2] / "log.md"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(console)
        root.addHandler(file_handler)

    root.info("Logging to %s (overwrite)", log_file)
    return root
