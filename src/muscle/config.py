"""Unified JSON config loader for VMuscle."""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

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




