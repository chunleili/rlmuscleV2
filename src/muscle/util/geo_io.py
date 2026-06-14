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


def _decode_bool_rle(unordered) -> np.ndarray:
    """Decode Houdini ['boolRLE', [count, bool, count, bool, ...]] to int32 indices."""
    if not unordered or unordered[0] != "boolRLE":
        return np.zeros(0, dtype=np.int32)
    runs = unordered[1]
    out = []
    pos = 0
    for k in range(0, len(runs), 2):
        count = int(runs[k])
        val = bool(runs[k + 1])
        if val:
            out.extend(range(pos, pos + count))
        pos += count
    return np.asarray(out, dtype=np.int32)


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
            if name not in ("P", "gluetoanimation", "pts", "target_pos"):
                continue
            if "values" not in data:
                continue
            values = data["values"]

            if name == "P":
                positions = np.asarray(values[5], dtype=np.float32)
            elif name == "gluetoanimation":
                gluetoanimation = np.asarray(values[5][0], dtype=np.bool_)
            elif name == "pts":
                pts = np.asarray(values[5][0], dtype=np.int32)
            elif name == "target_pos":
                target_pos = np.asarray(values[5], dtype=np.float32)

        # Generic named point-attribute capture.
        self._attrs: dict[str, np.ndarray] = {}
        for entry in pointattrs:
            meta = _pair_list_to_dict(entry[0])
            data = _pair_list_to_dict(entry[1])
            name = meta["name"]
            if "values" not in data:
                continue
            payload = np.asarray(data["values"][5])
            if payload.ndim == 2 and payload.shape[0] == 1:   # scalar attr stored (1,NV)
                payload = payload[0]
            self._attrs[name] = payload

        # Primitive attributes (per-tet).  String attrs (e.g. muscle_id) store a
        # `strings` table + a per-primitive integer index into it; numeric attrs
        # use the same values[5] payload as point attrs.
        self._prim_attrs: dict[str, np.ndarray] = {}
        self._prim_attr_strings: dict[str, list] = {}
        for entry in attrs.get("primitiveattributes", []):
            meta = _pair_list_to_dict(entry[0])
            data = _pair_list_to_dict(entry[1])
            name = meta["name"]
            if "strings" in data and "indices" in data:
                idxd = _pair_list_to_dict(data["indices"])
                arr = idxd.get("arrays")
                if arr is not None:
                    self._prim_attrs[name] = np.asarray(arr).ravel().astype(np.int32)
                    self._prim_attr_strings[name] = list(data["strings"])
            elif "values" in data:
                payload = np.asarray(data["values"][5])
                if payload.ndim == 2 and payload.shape[0] == 1:
                    payload = payload[0]
                self._prim_attrs[name] = payload

        # Topology / tet indices
        topo = _pair_list_to_dict(root["topology"])
        pointref = _pair_list_to_dict(topo["pointref"])
        indices = np.asarray(pointref["indices"], dtype=np.int32)

        n_tets = 0
        prims = root.get("primitives", [])
        for blk in prims:
            hdr = _pair_list_to_dict(blk[0])
            if hdr.get("type") == "Tetrahedron_run":
                body = _pair_list_to_dict(blk[1])
                n_tets = int(body["nprimitives"])
                break
        if n_tets > 0:
            tet_indices = indices[: n_tets * 4].reshape(n_tets, 4)
        else:
            primitive_count = int(root.get("primitivecount", 0))
            if primitive_count > 0 and indices.size:
                vpp = indices.size // primitive_count
                tet_indices = indices.reshape(primitive_count, vpp)
            else:
                tet_indices = indices.reshape(0, 0)

        # Point groups.
        self._groups: dict[str, np.ndarray] = {}
        for grp in root.get("pointgroups", []):
            gmeta = _pair_list_to_dict(grp[0])
            gsel = _pair_list_to_dict(grp[1])["selection"]
            seld = _pair_list_to_dict(gsel)
            if "ordered" in seld:
                idx = np.asarray(seld["ordered"], dtype=np.int32)
            elif "unordered" in seld:
                idx = _decode_bool_rle(seld["unordered"])
            else:
                continue
            self._groups[gmeta["name"]] = idx

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

    def point_attr(self, name: str) -> np.ndarray:
        if name not in self._attrs:
            raise KeyError(f"point attribute '{name}' not in {sorted(self._attrs)}")
        return self._attrs[name]

    def has_point_attr(self, name: str) -> bool:
        return name in self._attrs

    def point_group(self, name: str) -> np.ndarray:
        if name not in self._groups:
            raise KeyError(f"point group '{name}' not in {sorted(self._groups)}")
        return self._groups[name]

    def has_prim_attr(self, name: str) -> bool:
        return name in self._prim_attrs

    def prim_attr(self, name: str) -> np.ndarray:
        """Per-primitive attribute. For string attrs this is the integer index
        into :meth:`prim_attr_strings` (one int per tet)."""
        if name not in self._prim_attrs:
            raise KeyError(f"prim attribute '{name}' not in {sorted(self._prim_attrs)}")
        return self._prim_attrs[name]

    def prim_attr_strings(self, name: str) -> list:
        """String table for a string-typed prim attr (index -> name)."""
        return self._prim_attr_strings.get(name, [])


def _find_point_attr(pointattrs, name):
    """Return the [meta, data] entry for point attribute *name*, or None."""
    for entry in pointattrs:
        if _pair_list_to_dict(entry[0]).get("name") == name:
            return entry
    return None


def _set_pairlist_value(pairlist, key, value):
    """Set ``pairlist[i+1] = value`` for the flat [k0,v0,...] where vk==key."""
    for i in range(0, len(pairlist), 2):
        if pairlist[i] == key:
            pairlist[i + 1] = value
            return True
    return False


def _make_int_point_attr(name, values):
    """Build a Houdini int32 scalar point-attribute entry [meta, data]."""
    vals = [int(v) for v in values]
    meta = ["scope", "public", "type", "numeric", "name", name, "options", {}]
    data = [
        "size", 1, "storage", "int32",
        "defaults", ["size", 1, "storage", "int32", "values", [0]],
        "values", ["size", 1, "storage", "int32", "arrays", [vals]],
    ]
    return [meta, data]


def _make_float_point_attr(name, values):
    """Build a Houdini fpreal32 scalar point-attribute entry [meta, data]."""
    vals = [float(v) for v in values]
    meta = ["scope", "public", "type", "numeric", "name", name, "options", {}]
    data = [
        "size", 1, "storage", "fpreal32",
        "defaults", ["size", 1, "storage", "fpreal64", "values", [0]],
        "values", ["size", 1, "storage", "fpreal32", "arrays", [vals]],
    ]
    return [meta, data]


def _upsert_scalar_attr(pointattrs, name, values, npoints, maker):
    """Create *name* point attr if absent, else overwrite its array values."""
    arr = np.asarray(values)
    if arr.shape[0] != npoints:
        raise ValueError(f"{name} length {arr.shape[0]} != pointcount {npoints}")
    entry = _find_point_attr(pointattrs, name)
    if entry is None:
        pointattrs.append(maker(name, arr))
    else:
        vvals = _pair_list_to_dict(entry[1])["values"]
        _set_pairlist_value(vvals, "arrays", [arr.tolist()])


class GeoWriter:
    """In-place .geo writer: parse a template once, then rewrite per frame.

    Everything in the template (topology, primitives, groups, all other point
    attributes) is preserved verbatim. Each :meth:`write` only overwrites the
    ``P`` position tuples and an optional integer ``is_collide`` point attribute
    (created if the template lacks it, overwritten if present). This keeps the
    asset a faithful Houdini round-trip — load it back and only P / is_collide
    have changed.
    """

    def __init__(self, template_path: str):
        with open(template_path, "r") as fp:
            self._root = json.load(fp)
        root = _pair_list_to_dict(self._root)
        self._npoints = int(root["pointcount"])
        attrs = _pair_list_to_dict(root["attributes"])
        self._pointattrs = attrs["pointattributes"]   # live ref into self._root

        p_entry = _find_point_attr(self._pointattrs, "P")
        if p_entry is None:
            raise KeyError("template .geo has no 'P' attribute")
        self._p_values = _pair_list_to_dict(p_entry[1])["values"]  # flat list ref

    def write(self, out_path: str, positions: np.ndarray,
              is_collide: np.ndarray | None = None,
              pen_depth: np.ndarray | None = None):
        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape[0] != self._npoints:
            raise ValueError(
                f"positions has {pos.shape[0]} points, template has "
                f"{self._npoints}")
        _set_pairlist_value(self._p_values, "tuples", pos.tolist())

        if is_collide is not None:
            _upsert_scalar_attr(self._pointattrs, "is_collide",
                                np.asarray(is_collide).astype(np.int32),
                                self._npoints, _make_int_point_attr)
        if pen_depth is not None:
            _upsert_scalar_attr(self._pointattrs, "pen_depth",
                                np.asarray(pen_depth).astype(np.float32),
                                self._npoints, _make_float_point_attr)

        with open(out_path, "w") as fp:
            json.dump(self._root, fp)


def read_geo_attr_cached(geo_path: str, attr: str) -> np.ndarray:
    """Return one point attribute from a `.geo`, cached as binary `.npy`.

    Houdini `.geo` files are large JSON (a cons frame is ~20 MB / 157k points);
    `json.load` dominates per-frame cost in animated sims. We cache the single
    extracted array next to the source as ``_npy_cache/<stem>_<attr>.npy`` and
    reload that on subsequent calls, guarded by mtime so an edited `.geo`
    re-parses. ``attr`` is ``"P"`` (positions) or ``"target_pos"``.
    """
    geo = Path(geo_path)
    cache = geo.parent / "_npy_cache" / f"{geo.stem}_{attr}.npy"
    if cache.exists() and cache.stat().st_mtime >= geo.stat().st_mtime:
        return np.load(cache)
    g = GeoFile(str(geo))
    arr = g.positions if attr == "P" else getattr(g, attr)
    cache.parent.mkdir(exist_ok=True)
    # atomic write so a killed run can't leave a truncated cache file.
    # tmp must end in .npy or np.save appends a second .npy to the name.
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(cache)
    return arr


def read_cons_frame(geo_dir: str, frame: int) -> np.ndarray:
    """Return the (NCONS,3) target_pos array from `cons_<frame>.geo` (cached)."""
    return read_geo_attr_cached(str(Path(geo_dir) / f"cons_{frame}.geo"), "target_pos")
