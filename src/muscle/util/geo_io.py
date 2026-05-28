"""Houdini JSON .geo reader, trimmed to what the lion ARAP-HPBD sim needs.

Mirrors `realtimeMuscle/engine/geo.py` for the fields the lion solver
consumes: P (positions), tet vertex indices, gluetoanimation (bool[NV]),
pts (int[NCONS]) and target_pos (float32[NCONS,3]).

The file is a pair-list JSON `[k0, v0, k1, v1, ...]`. Each point attribute
is a `[metadata_pairlist, data_pairlist]` pair; the data pair-list always
exposes a `values` field whose 6th positional element (`values[5]`) holds
the actual numeric payload — vector attributes (P, target_pos) put the
(N,3) tuples there directly; scalar attributes (gluetoanimation, pts) wrap
the array in one extra level, so the payload is `values[5][0]`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pair_list_to_dict(pairs):
    return dict(zip(pairs[0::2], pairs[1::2]))


class GeoFile:
    def __init__(self, path: str):
        with open(path, "r") as fp:
            raw = json.load(fp)

        root = _pair_list_to_dict(raw)
        attrs = _pair_list_to_dict(root["attributes"])
        pointattrs = attrs.get("pointattributes", [])

        positions = None
        gluetoanimation = None
        pts = None
        target_pos = None

        for entry in pointattrs:
            meta = _pair_list_to_dict(entry[0])
            data = _pair_list_to_dict(entry[1])
            name = meta["name"]
            values = data["values"]

            if name == "P":
                positions = np.asarray(values[5], dtype=np.float32)
            elif name == "gluetoanimation":
                gluetoanimation = np.asarray(values[5][0], dtype=np.bool_)
            elif name == "pts":
                pts = np.asarray(values[5][0], dtype=np.int32)
            elif name == "target_pos":
                target_pos = np.asarray(values[5], dtype=np.float32)

        # Topology / tet indices
        topo = _pair_list_to_dict(root["topology"])
        pointref = _pair_list_to_dict(topo["pointref"])
        indices = np.asarray(pointref["indices"], dtype=np.int32)
        primitive_count = int(root.get("primitivecount", 0))

        if primitive_count > 0 and indices.size:
            verts_per_prim = indices.size // primitive_count
            tet_indices = indices.reshape(primitive_count, verts_per_prim)
        else:
            tet_indices = indices.reshape(0, 0)

        self._positions = positions
        self._tet_indices = tet_indices
        self._gluetoanimation = gluetoanimation
        self._pts = pts
        self._target_pos = target_pos

    @property
    def positions(self) -> np.ndarray:
        if self._positions is None:
            raise KeyError("attribute 'P' not present in this .geo")
        return self._positions

    @property
    def tet_indices(self) -> np.ndarray:
        return self._tet_indices

    @property
    def gluetoanimation(self) -> np.ndarray:
        # Matches realtimeMuscle/engine/geo.py:get_gluetoaniamtion which fills a
        # zero array when the attribute is absent (the case for lion/restpos.geo).
        if self._gluetoanimation is None:
            n = 0 if self._positions is None else self._positions.shape[0]
            self._gluetoanimation = np.zeros(n, dtype=np.bool_)
        return self._gluetoanimation

    @property
    def pts(self) -> np.ndarray:
        if self._pts is None:
            raise KeyError("attribute 'pts' not present in this .geo")
        return self._pts

    @property
    def target_pos(self) -> np.ndarray:
        if self._target_pos is None:
            raise KeyError("attribute 'target_pos' not present in this .geo")
        return self._target_pos


def read_cons_frame(geo_dir: str, frame: int) -> np.ndarray:
    """Return the (NCONS,3) target_pos array from `cons_<frame>.geo`.

    Parses the whole file each call — fine for the lion sim's 90 frames.
    """
    return GeoFile(str(Path(geo_dir) / f"cons_{frame}.geo")).target_pos
