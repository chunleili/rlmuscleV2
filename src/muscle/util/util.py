"""Centralized config loader and logging setup for VMuscle."""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: "str | Path | None" = None,
    level: int = logging.DEBUG,
):
    """Configure the 'VMuscle' logger hierarchy and root-level console output."""
    if log_file is None:
        log_file = Path(__file__).resolve().parents[3] / "log.md"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(console)
        root_logger.addHandler(file_handler)

    root_logger.info("Logging to %s (overwrite)", log_file)
    return root_logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Fields that should be converted to Path objects
_PATH_FIELDS = frozenset(["geo_path", "bone_geo_path", "ground_mesh_path", "coord_mesh_path"])

# SimConfig is a SimpleNamespace — all values come from JSON
SimConfig = SimpleNamespace


def load_config(json_path, args=None) -> SimpleNamespace:
    """Load JSON config. If args is given, override its attributes; otherwise create new."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"json file {json_path} not exist!")
    logger.info("Using json config: %s", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in _PATH_FIELDS:
        if key in data:
            data[key] = Path(data[key])
    if "coupling" in data:
        data["coupling"] = SimpleNamespace(**data["coupling"])

    if args is None:
        return SimConfig(**data)
    for key, value in data.items():
        setattr(args, key, value)
    return args



def _check_unzipped(path: str) -> None:
    """Raise if a required .geo asset is missing — model dirs ship as zips.

    Each case's geometry (restpos, restpos_lowres, cons_*.geo) lives inside
    ``data/<case>/model.zip`` to save LFS storage. Unzip once before running
    any sim. See data/README.md.
    """
    p = Path(path)
    if p.exists():
        return
    parts = p.parts
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            case_dir = Path(*parts[: i + 2])
            zip_path = case_dir / "model.zip"
            if zip_path.exists():
                raise FileNotFoundError(
                    f"missing asset: {path}\n"
                    f"model assets are zipped — extract first:\n"
                    f"    (cd {case_dir} && unzip -o model.zip)\n"
                    f"see data/README.md"
                )
    raise FileNotFoundError(f"missing asset: {path}")