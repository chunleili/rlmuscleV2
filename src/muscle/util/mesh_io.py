"""Mesh I/O for RLMuscle — .usd and .geo formats only.

Provides:
- load_mesh(path)       — muscle TetMesh (positions, tets, fibers, tendon_mask, geo)
- load_bone_mesh(path)  — bone surface mesh (geo, positions, indices, muscle_ids)
- build_surface_tris()  — extract boundary triangles from tetrahedra
- save_ply()            — export surface mesh to PLY
- UsdTetExporter        — export tet mesh animation to USD
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Hardcoded Y-up -> Z-up rotation: 90 deg around X axis.
_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Surface triangle extraction
# ---------------------------------------------------------------------------

def build_surface_tris(tets: np.ndarray, positions: np.ndarray = None) -> np.ndarray:
    """Extract boundary triangles from tetrahedra.

    If *positions* is given, orient each triangle so its outward normal points
    away from the tet interior (needed for correct backface culling).
    """
    tet_faces = (
        (1, 2, 3, 0),
        (0, 3, 2, 1),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
    )
    counts: dict[tuple, list] = {}
    for tet in tets:
        for f in tet_faces:
            tri = (int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]]))
            opp = int(tet[f[3]])
            key = tuple(sorted(tri))
            counts.setdefault(key, []).append((tri, opp))

    surface = []
    for entries in counts.values():
        if len(entries) == 1:
            tri, opp = entries[0]
            if positions is not None:
                p0, p1, p2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]
                n = np.cross(p1 - p0, p2 - p0)
                d = positions[opp] - p0
                if np.dot(n, d) > 0:
                    tri = (tri[0], tri[2], tri[1])  # flip winding
            surface.append(tri)
    return np.asarray(surface, dtype=np.int32)


def vectorized_surface_tris(tets: np.ndarray) -> np.ndarray:
    """Extract boundary tris from a tet mesh — fast vectorized version.

    A face is on the surface iff it appears in exactly one tet. Faces are
    canonicalized by sorting their 3 vertex ids; the original (oriented)
    triangle is recovered by looking up which tet+face slot produced it.
    """
    NT = tets.shape[0]
    # Each tet has 4 faces; use the standard outward winding convention.
    face_slots = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)
    tris = tets[:, face_slots].reshape(-1, 3)        # (4*NT, 3) oriented
    keys = np.sort(tris, axis=1)                      # canonical key

    # Find unique canonical faces and how many tets share each.
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inv] == 1
    return tris[boundary_mask].astype(np.int32)


# ---------------------------------------------------------------------------
# Internal: format detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal: USD helpers
# ---------------------------------------------------------------------------

def _read_primvar(pv_api, name, dtype=np.float32):
    """Read a named primvar, return numpy array or None."""
    pv = pv_api.GetPrimvar(name)
    if pv and pv.Get() is not None:
        return np.asarray(pv.Get(), dtype=dtype)
    return None


def _load_mesh_usd(path: Path, y_up_to_z_up: bool = False, prim_path: str = None):
    """Load muscle TetMesh from USD.

    Returns (positions, tets, fibers, tendon_mask, geo_namespace).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {path}")

    tet_prim = None
    if prim_path:
        p = stage.GetPrimAtPath(prim_path)
        if not p or not p.IsValid() or not p.IsA(UsdGeom.TetMesh):
            raise ValueError(f"Prim {prim_path} not found or not a TetMesh in {path}")
        tet_prim = p
    else:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.TetMesh):
                tet_prim = prim
                break
    if tet_prim is None:
        raise ValueError(f"No TetMesh prim found in {path}")

    tm = UsdGeom.TetMesh(tet_prim)
    positions = np.asarray(tm.GetPointsAttr().Get(), dtype=np.float32)
    tets = np.asarray(tm.GetTetVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 4)

    pv_api = UsdGeom.PrimvarsAPI(tet_prim)
    fibers = _read_primvar(pv_api, "materialW", np.float32)
    tendon_mask = _read_primvar(pv_api, "tendonmask", np.float32)

    if y_up_to_z_up:
        positions = (positions @ _Y_TO_Z.T).astype(np.float32)
        if fibers is not None and fibers.ndim == 2 and fibers.shape[1] == 3:
            fibers = (fibers @ _Y_TO_Z.T).astype(np.float32)

    # Build a geo-like namespace so constraint code can do getattr(self.geo, mask_name)
    geo = SimpleNamespace()
    geo.positions = positions.tolist()
    geo.vert = tets.tolist()
    geo.pointattr = {}

    for pv in pv_api.GetPrimvars():
        name = pv.GetName().replace("primvars:", "")
        if pv.GetInterpolation() != "vertex":
            continue
        val = pv.Get()
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim in (1, 2):
            setattr(geo, name, arr.tolist())
            geo.pointattr[name] = arr.tolist()

    print(f"Loaded USD TetMesh from {path}: {positions.shape[0]} verts, {tets.shape[0]} tets")
    return positions, tets, fibers, tendon_mask, geo


def _load_mesh_usd_multi(path: Path, prim_root: str):
    """Load every UsdGeom.TetMesh under *prim_root* and concatenate into one mesh.

    Each TetMesh contributes its own vertices (transformed to world space),
    tet indices (offset to the merged vertex range) and per-vertex primvars
    (concatenated; missing per-muscle primvars are zero-padded so the merged
    array length matches total verts).

    Vector primvars whose name is ``materialW`` are rotated by the prim's
    local-to-world rotation; other primvars are concatenated as-is.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {path}")
    root = stage.GetPrimAtPath(prim_root)
    if not root or not root.IsValid():
        raise ValueError(f"Prim root {prim_root} not in {path}")

    rest_t = stage.GetStartTimeCode()
    if rest_t == Usd.TimeCode.Default().GetValue():
        rest_t = 1.0
    xcache = UsdGeom.XformCache(rest_t)

    all_pos: list[np.ndarray] = []
    all_tets: list[np.ndarray] = []
    pv_segments: dict[str, list[tuple[int, np.ndarray]]] = {}  # name -> [(seg_idx, arr)]
    seg_lens: list[int] = []
    seg_idx = 0
    offset = 0
    n_loaded = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.TetMesh):
            continue
        tm = UsdGeom.TetMesh(prim)
        pts_raw = tm.GetPointsAttr().Get()
        tets_raw = tm.GetTetVertexIndicesAttr().Get()
        if pts_raw is None or tets_raw is None:
            continue
        pts_local = np.asarray(pts_raw, dtype=np.float32)
        tets_local = np.asarray(tets_raw, dtype=np.int32).reshape(-1, 4)

        wm = xcache.GetLocalToWorldTransform(prim)
        R = np.array([[wm[i][j] for j in range(3)] for i in range(3)], dtype=np.float32)
        T = np.array([wm[3][j] for j in range(3)], dtype=np.float32)
        pts_world = (pts_local @ R + T).astype(np.float32)

        n_pts = len(pts_world)
        all_pos.append(pts_world)
        all_tets.append((tets_local + offset).astype(np.int32))

        pv_api = UsdGeom.PrimvarsAPI(prim)
        for pv in pv_api.GetPrimvars():
            if pv.GetInterpolation() != "vertex":
                continue
            val = pv.Get()
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.ndim not in (1, 2):
                continue
            if arr.shape[0] != n_pts:
                continue  # length mismatch — ignore quietly
            name = pv.GetName().replace("primvars:", "")
            if name == "materialW" and arr.ndim == 2 and arr.shape[1] == 3:
                arr = (arr.astype(np.float32) @ R).astype(arr.dtype)
            pv_segments.setdefault(name, []).append((seg_idx, arr))

        seg_lens.append(n_pts)
        offset += n_pts
        seg_idx += 1
        n_loaded += 1

    if n_loaded == 0:
        raise ValueError(f"No TetMesh prims found under {prim_root}")

    positions = np.vstack(all_pos).astype(np.float32)
    tets = np.vstack(all_tets).astype(np.int32)
    total_v = positions.shape[0]

    geo = SimpleNamespace()
    geo.positions = positions.tolist()
    geo.vert = tets.tolist()
    geo.pointattr = {}

    fibers = None
    tendon_mask = None
    for name, segs in pv_segments.items():
        sample = segs[0][1]
        if sample.ndim == 1:
            full = np.zeros((total_v,), dtype=sample.dtype)
        else:
            full = np.zeros((total_v, sample.shape[1]), dtype=sample.dtype)
        for s_idx, arr in segs:
            start = sum(seg_lens[:s_idx])
            full[start:start + seg_lens[s_idx]] = arr
        setattr(geo, name, full.tolist())
        geo.pointattr[name] = full.tolist()
        if name == "materialW":
            fibers = full.astype(np.float32)
        elif name == "tendonmask":
            tendon_mask = full.astype(np.float32)

    print(f"Loaded {n_loaded} TetMesh prims under {prim_root}: "
          f"{total_v} verts, {tets.shape[0]} tets")
    return positions, tets, fibers, tendon_mask, geo


def _load_bone_usd(usd_path: Path, bone_root: str = "/character/bone"):
    """Load bone mesh from USD (multiple Mesh prims under *bone_root*).

    Vertex positions are returned in **world space** (parent xforms applied)
    so they live in the same coordinate frame as muscle meshes loaded by
    ``_load_mesh_usd``.

    Returns (None, positions, indices, muscle_ids).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")

    root = stage.GetPrimAtPath(bone_root)
    if not root or not root.IsValid():
        return None, np.zeros((0, 3), np.float32), np.zeros(0, np.int32), {}

    rest_t = stage.GetStartTimeCode()
    if rest_t == Usd.TimeCode.Default().GetValue():
        rest_t = 1.0
    xcache = UsdGeom.XformCache(rest_t)

    all_pos, all_idx, all_mid = [], [], []
    offset = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts_raw = m.GetPointsAttr().Get()
        if pts_raw is None:
            continue
        pts_local = np.asarray(pts_raw, dtype=np.float32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        # Apply mesh's local-to-world (row-vector convention).
        wm = xcache.GetLocalToWorldTransform(prim)
        R = np.array([[wm[i][j] for j in range(3)] for i in range(3)], dtype=np.float32)
        T = np.array([wm[3][j] for j in range(3)], dtype=np.float32)
        pts = (pts_local @ R + T).astype(np.float32)

        parent = prim.GetParent()
        bone_name = str(parent.GetName()) if parent and parent.IsValid() else str(prim.GetName())

        all_pos.append(pts)
        all_idx.append(idx + offset)
        all_mid.extend([bone_name] * len(pts))
        offset += len(pts)

    if len(all_pos) == 0:
        return None, np.zeros((0, 3), np.float32), np.zeros(0, np.int32), {}

    positions = np.vstack(all_pos).astype(np.float32)
    indices = np.concatenate(all_idx).astype(np.int32)
    muscle_ids = _build_muscle_id_mapping(all_mid)
    return None, positions, indices, muscle_ids


def _build_muscle_id_mapping(muscle_ids_list):
    """Build dict mapping muscle_id -> vertex index array from per-vertex id list."""
    mapping = {}
    for v_idx, mid in enumerate(muscle_ids_list):
        if mid not in mapping:
            mapping[mid] = []
        mapping[mid].append(v_idx)
    for mid in mapping:
        mapping[mid] = np.array(mapping[mid], dtype=np.int32)
    print(f"Bone muscle_id groups: {list(mapping.keys())}")
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mesh(path: Path, prim_path: str = None, prim_root: str = None):
    """Load muscle TetMesh(es) from USD.

    Returns (positions, tets, fibers, tendon_mask, geo).

    - If *prim_root* is given, every TetMesh under that prim is loaded and
      concatenated into one merged mesh (vertices in world space, tet
      indices offset; per-vertex primvars stitched together).
    - Else if *prim_path* is given, only that single TetMesh is loaded.
    - Else the first TetMesh in the stage is loaded.
    """
    if path is None:
        raise ValueError("No mesh path provided.")
    if prim_root:
        return _load_mesh_usd_multi(path, prim_root=prim_root)
    return _load_mesh_usd(path, y_up_to_z_up=False, prim_path=prim_path)


def load_bone_mesh(path: Path, bone_root: str = "/character/bone"):
    """Load bone surface mesh from USD.

    Returns (bone_geo, bone_pos, bone_indices, bone_muscle_ids) where:
    - bone_geo: always None (legacy slot, kept for tuple stability)
    - bone_pos: np.ndarray shape (N, 3)
    - bone_indices: np.ndarray flat int32
    - bone_muscle_ids: dict mapping muscle_id -> vertex index array, derived
      from prim hierarchy under bone_root/<bone_name>/Mesh.
    """
    return _load_bone_usd(path, bone_root=bone_root)


class UsdTetRest:
    """GeoFile-compatible rest reader for a USD ``TetMesh`` muscle asset.

    Exposes only the slice of the ``GeoFile`` API that ``MultiGridSim`` reads at
    build time: ``positions``, ``tet_indices``, ``point_attr`` /
    ``has_point_attr`` (materialW fiber dir, tendonmask, optional Cd) and the
    ``prim_attr`` family (muscle_id — usually absent on a TetMesh, reported
    missing). Points are pushed to world space; ``materialW`` is a direction so
    it gets the rotation only (no translation).

    Winding is normalised to the solver's convention: the ARAP rest frame
    expects ``det(Dm) < 0`` == USD ``orientation = leftHanded`` (the winding
    Houdini ``.geo`` assets carry — verified against the working humanV2 rest).
    The decision is read from the ``orientation`` TOKEN, not per-tet det: a
    ``rightHanded`` mesh (Emil) is flipped whole, a ``leftHanded`` mesh (otto)
    is left as-is. Keying off the token (vs per-tet ``det>0``) avoids mis-flipping
    sliver tets whose computed det is float-noise near zero. A reflecting world
    xform (det(R)<0) inverts the effective handedness and is accounted for.
    """

    def __init__(self, usd_path, prim_path: str | None = None,
                 prim_root: str | None = None, local_space: bool = False,
                 scale: float = 1.0):
        from pxr import Usd, UsdGeom

        self._scale = float(scale)
        self._local_space = bool(local_space)
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise FileNotFoundError(f"Cannot open muscle USD: {usd_path}")

        # Collect the TetMesh prim(s): every TetMesh under `prim_root` (merge), or
        # a single `prim_path` (or the first found).
        prims = []
        if prim_root is not None:
            root = stage.GetPrimAtPath(prim_root)
            if not root or not root.IsValid():
                raise ValueError(f"prim_root {prim_root} not in {usd_path}")
            prims = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.TetMesh)]
        else:
            for p in stage.Traverse():
                if p.IsA(UsdGeom.TetMesh) and (
                        prim_path is None or str(p.GetPath()) == prim_path):
                    prims = [p]
                    break
        if not prims:
            raise ValueError(
                f"no TetMesh prim in {usd_path}"
                + (f" under {prim_root}" if prim_root else
                   f" at {prim_path}" if prim_path else ""))

        rest_t = stage.GetStartTimeCode()
        if rest_t == Usd.TimeCode.Default().GetValue():
            rest_t = 1.0
        xcache = UsdGeom.XformCache(float(rest_t))

        all_pos: list[np.ndarray] = []
        all_tets: list[np.ndarray] = []
        all_surf: list[np.ndarray] = []
        all_tet_mid: list[np.ndarray] = []    # per-tet authored muscle_id (global)
        mid_names: list = []                  # global muscle_id name table (shared)
        pv_segments: dict[str, list] = {}     # name -> [(seg_idx, arr)]
        seg_lens: list[int] = []
        n_flipped = 0
        v_off = 0
        for seg_idx, prim in enumerate(prims):
            pts = prim.GetAttribute("points").Get()
            tvi = prim.GetAttribute("tetVertexIndices").Get()
            if pts is None or tvi is None:
                raise ValueError(f"{prim.GetPath()} missing points / tetVertexIndices")
            pl = np.asarray(pts, np.float32)
            tl = np.asarray(tvi, np.int64).reshape(-1, 4)

            if self._local_space:            # ignore viz xforms: raw local geometry
                R = np.eye(3, dtype=np.float32)
                pw = pl
            else:                            # push to world (Emil: /char cm->m scale)
                wm = xcache.GetLocalToWorldTransform(prim)
                R = np.array([[wm[i][j] for j in range(3)] for i in range(3)], np.float32)
                T = np.array([wm[3][j] for j in range(3)], np.float32)
                pw = (pl @ R + T).astype(np.float32)
            if self._scale != 1.0:
                pw = (pw * np.float32(self._scale)).astype(np.float32)

            # Winding from the orientation TOKEN (not per-tet det); reflecting xform
            # swaps handedness. Solver ARAP wants leftHanded / det<0.
            orient = prim.GetAttribute("orientation").Get() or "rightHanded"
            reflect = bool(np.linalg.det(R.astype(np.float64)) < 0)
            if (orient == "rightHanded") ^ reflect:
                tl = tl[:, [0, 2, 1, 3]]     # swap v1,v2 -> negate det
                n_flipped += 1

            all_pos.append(pw)
            all_tets.append((tl + v_off).astype(np.int64))
            sfvi = prim.GetAttribute("surfaceFaceVertexIndices").Get()
            if sfvi is not None:
                all_surf.append((np.asarray(sfvi, np.int64) + v_off))

            n_pts = len(pw)
            pv_api = UsdGeom.PrimvarsAPI(prim)
            for pv in pv_api.GetPrimvars():
                if pv.GetInterpolation() != "vertex":
                    continue
                val = pv.Get()
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.ndim not in (1, 2) or arr.shape[0] != n_pts:
                    continue
                if arr.dtype.kind not in "fiu":   # skip strings etc.
                    continue
                name = pv.GetName().replace("primvars:", "")
                arr = arr.astype(np.float32)
                if name == "materialW" and arr.ndim == 2 and arr.shape[1] == 3:
                    arr = arr @ R           # direction: rotate only (R=I if local)
                pv_segments.setdefault(name, []).append((seg_idx, arr))

            # Authored per-tet muscle_id (uniform, indexed into a global name
            # table shared by all prims; bilateral prims carry both L/R ids).
            # Indices ARE the global per-tet id. Winding-flip swaps tet columns,
            # not rows, so the id order still aligns. Fall back to a per-prim id
            # (negative, distinct from any real id) when the primvar is absent.
            mid_pv = pv_api.GetPrimvar("muscle_id")
            mid_idx = mid_pv.GetIndices() if (mid_pv and mid_pv.IsIndexed()) else None
            if mid_idx is not None and len(mid_idx) == tl.shape[0]:
                all_tet_mid.append(np.asarray(mid_idx, np.int64))
                if not mid_names:                      # global table, captured once
                    mid_names = [str(x) for x in (mid_pv.Get() or [])]
            else:
                all_tet_mid.append(np.full(tl.shape[0], -1 - seg_idx, np.int64))

            seg_lens.append(n_pts)
            v_off += n_pts

        self._positions = np.ascontiguousarray(np.vstack(all_pos).astype(np.float32))
        self._tet_indices = np.ascontiguousarray(np.vstack(all_tets).astype(np.int64))
        self._surface_tris = (np.concatenate(all_surf).astype(np.int64)
                              if all_surf else None)
        total_v = self._positions.shape[0]
        # Per-tet + per-vertex muscle id (authored muscle_id, finer than per-prim:
        # bilateral prims split L/R). Scatter each tet's id onto its 4 verts; lets
        # the muscle-to-muscle glue pair only ACROSS different muscles.
        self._tet_muscle_id = np.concatenate(all_tet_mid).astype(np.int64)
        vid = np.full(total_v, 0, np.int64)
        for k in range(4):
            vid[self._tet_indices[:, k]] = self._tet_muscle_id
        self._vert_muscle_id = vid

        # Merge per-vertex primvars (zero-pad muscles missing a given primvar).
        self._attrs = {}
        seg_start = np.concatenate([[0], np.cumsum(seg_lens)])
        for name, segs in pv_segments.items():
            sample = segs[0][1]
            shape = (total_v,) if sample.ndim == 1 else (total_v, sample.shape[1])
            full = np.zeros(shape, dtype=np.float32)
            for s_idx, arr in segs:
                full[seg_start[s_idx]:seg_start[s_idx] + seg_lens[s_idx]] = arr
            self._attrs[name] = full
        # Per-tet muscle_id exposed as a "prim attr" (name table = global string
        # list) so name-based muscle selection (activation, coloring) works.
        if mid_names:
            self._prim_attrs: dict = {"muscle_id": self._tet_muscle_id}
            self._prim_attr_strings: dict = {"muscle_id": mid_names}
        else:
            self._prim_attrs = {}
            self._prim_attr_strings = {}

        det = np.einsum("ij,ij->i",
                        np.cross(self._positions[self._tet_indices[:, 1]] - self._positions[self._tet_indices[:, 0]],
                                 self._positions[self._tet_indices[:, 2]] - self._positions[self._tet_indices[:, 0]]),
                        self._positions[self._tet_indices[:, 3]] - self._positions[self._tet_indices[:, 0]])
        print(f"[muscle-usd] {usd_path}: {len(prims)} prim(s) "
              f"({'local' if self._local_space else 'world'}), "
              f"{total_v} verts, {self._tet_indices.shape[0]} tets "
              f"(flipped {n_flipped}/{len(prims)}, det>0 left {int((det > 0).sum())}); "
              f"surf_tris={0 if self._surface_tris is None else len(self._surface_tris)//3}; "
              f"attrs={sorted(self._attrs)}")

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def tet_indices(self) -> np.ndarray:
        return self._tet_indices

    @property
    def vert_muscle_id(self) -> np.ndarray:
        """Per-vertex source-prim (muscle) id, 0..n_prims-1."""
        return self._vert_muscle_id

    def point_attr(self, name: str) -> np.ndarray:
        if name not in self._attrs:
            raise KeyError(f"point attr '{name}' not in {sorted(self._attrs)}")
        return self._attrs[name]

    def has_point_attr(self, name: str) -> bool:
        return name in self._attrs

    def has_prim_attr(self, name: str) -> bool:
        return name in self._prim_attrs

    def prim_attr(self, name: str) -> np.ndarray:
        if name not in self._prim_attrs:
            raise KeyError(f"prim attr '{name}' not present")
        return self._prim_attrs[name]

    def prim_attr_strings(self, name: str) -> list:
        return self._prim_attr_strings.get(name, [])

    @property
    def surface_tris(self) -> np.ndarray | None:
        """Authored boundary triangles (flat int, indexing positions), offset and
        concatenated across merged prims; None if the asset has no
        surfaceFaceVertexIndices."""
        return self._surface_tris

    def has_point_group(self, name: str) -> bool:
        return name == "surface_points" and self._surface_tris is not None

    def point_group(self, name: str) -> np.ndarray:
        """GeoFile-compatible: 'surface_points' = unique verts of the authored
        surface tris."""
        if name == "surface_points" and self._surface_tris is not None:
            return np.unique(self._surface_tris).astype(np.int64)
        raise KeyError(f"point group '{name}' not present")


# ---------------------------------------------------------------------------
# Export: PLY
# ---------------------------------------------------------------------------

def save_ply(filename: str, positions: np.ndarray, surface_faces):
    """Save mesh surface to PLY file.

    Args:
        filename: output .ply path
        positions: Nx3 vertex positions
        surface_faces: list/array of (i, j, k) triangle indices
    """
    try:
        import meshio
        meshio.Mesh(points=positions, cells=[("triangle", surface_faces)]).write(filename)
        return
    except ImportError:
        pass
    with open(filename, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(positions)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write(f"element face {len(surface_faces)}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for p in positions:
            fp.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in surface_faces:
            fp.write(f"3 {f[0]} {f[1]} {f[2]}\n")


# ---------------------------------------------------------------------------
# Export: USD TetMesh animation
# ---------------------------------------------------------------------------

class UsdTetExporter:
    """Export tet mesh animation to a USD file with per-frame vertex positions."""

    def __init__(self, tet_indices, usd_path="output/anim.usd", prim_path="/tet", fps=24, start_frame=0):
        from pxr import Usd, UsdGeom, Gf, Vt

        self.usd_path = usd_path
        self.prim_path = prim_path
        self.fps = fps

        self.stage = Usd.Stage.CreateNew(usd_path)
        self.stage.SetStartTimeCode(start_frame)
        self.stage.SetEndTimeCode(start_frame)
        self.stage.SetTimeCodesPerSecond(fps)

        self.tet = UsdGeom.TetMesh.Define(self.stage, prim_path)
        self.tet.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)

        tet_vec4 = Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in tet_indices])
        self.tet.GetTetVertexIndicesAttr().Set(tet_vec4)

        surface_faces = UsdGeom.TetMesh.ComputeSurfaceFaces(self.tet)
        self.tet.CreateSurfaceFaceVertexIndicesAttr().Set(surface_faces)
        self.points_attr = self.tet.GetPointsAttr()

    def save_frame(self, positions, frame, display_color=None):
        """Save vertex positions for a given frame, with optional display color.

        Parameters
        ----------
        positions : np.ndarray
            (N, 3) vertex positions.
        frame : int
            Frame number / time code.
        display_color : tuple or None
            Optional (r, g, b) float color for the mesh at this frame.
        """
        from pxr import Usd, Vt, Gf

        assert positions.shape[1] == 3, "Positions must be Nx3 array"
        pts = Vt.Vec3fArray.FromNumpy(positions)
        self.points_attr.Set(pts, Usd.TimeCode(frame))
        self.stage.SetEndTimeCode(frame)

        if display_color is not None:
            color_attr = self.tet.GetDisplayColorAttr()
            if not color_attr:
                color_attr = self.tet.CreateDisplayColorAttr()
            r, g, b = display_color
            color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(r, g, b)]),
                           Usd.TimeCode(frame))

    def finalize(self):
        """Write USD file to disk."""
        self.stage.GetRootLayer().Save()


class MeshExporter:
    """General mesh exporter supporting multiple formats (USD, PLY).

    Supports a primary tet mesh plus optional additional triangle meshes
    (e.g., bones, tendons). For USD, all meshes are written to the same
    stage as separate prims. For PLY, each mesh is written as separate
    per-frame files.

    Args:
        path: Output directory (PLY frames) or base name (USD gets .usd suffix).
        format: "ply" or "usd".
        tet_indices: (N, 4) int array of tet vertex indices.
        positions: (V, 3) float array of rest positions (used by PLY to build surface faces).
        fps: Frames per second (USD only).
        prim_path: USD prim path for the primary tet mesh (USD only).
    """
    def __init__(self, path="output/anim", format="ply", tet_indices=None, positions=None,
                 fps=24, prim_path="/tet"):
        import os
        self.format = format
        self.abspath = Path(path).absolute()
        self._extra_meshes = {}  # name -> {faces, usd_points_attr (usd only)}
        if format == "usd":
            # USD writes a single file "<path>.usd"; only the parent dir is needed.
            os.makedirs(self.abspath.parent, exist_ok=True)
            usd_path = str(self.abspath) + ".usd"
            self.usdexporter = UsdTetExporter(
                tet_indices=tet_indices, usd_path=usd_path,
                prim_path=prim_path, fps=fps)
        elif format == "ply":
            os.makedirs(self.abspath, exist_ok=True)
            if tet_indices is None:
                raise ValueError("tet_indices is required for PLY export.")
            self.surface_faces = build_surface_tris(tet_indices, positions)

    def add_mesh(self, name, faces):
        """Register an additional triangle mesh with fixed topology.

        Args:
            name: Unique name for this mesh (e.g., "humerus", "tendon_prox").
            faces: (F, 3) int array of triangle face indices.
        """
        import numpy as np
        faces = np.asarray(faces, dtype=np.int32)
        entry = {"faces": faces}
        if self.format == "usd":
            from pxr import UsdGeom, Vt
            stage = self.usdexporter.stage
            prim_path = f"/{name}"
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
            # Flat face-vertex indices and per-face vertex counts
            fvi = Vt.IntArray(faces.flatten().tolist())
            fvc = Vt.IntArray([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr().Set(fvi)
            mesh.CreateFaceVertexCountsAttr().Set(fvc)
            entry["usd_points_attr"] = mesh.GetPointsAttr()
        self._extra_meshes[name] = entry

    def save_frame(self, positions, frame, display_color=None):
        """Save the primary tet mesh positions for this frame.

        Parameters
        ----------
        positions : np.ndarray
            (V, 3) vertex positions.
        frame : int
            Frame number.
        display_color : tuple or None
            Optional (r, g, b) float color for the mesh at this frame (USD only).
        """
        if self.format == "usd":
            self.usdexporter.save_frame(positions, frame,
                                        display_color=display_color)
        elif self.format == "ply":
            filename = f"frame_{frame:04d}.ply"
            path = self.abspath / filename
            save_ply(str(path), positions, self.surface_faces)
            if frame % 10 == 0:
                print(f"Saved PLY frame {frame} to {filename}")

    def save_mesh_frame(self, name, positions, frame, display_color=None):
        """Save positions for a registered triangle mesh at this frame.

        Parameters
        ----------
        name : str
            Name passed to add_mesh().
        positions : np.ndarray
            (V, 3) float array of vertex positions.
        frame : int
            Frame number.
        display_color : tuple or None
            Optional (r, g, b) float color for the mesh at this frame (USD only).
        """
        entry = self._extra_meshes[name]
        if self.format == "usd":
            from pxr import Usd, Vt, Gf
            pts = Vt.Vec3fArray.FromNumpy(positions.astype("float32"))
            entry["usd_points_attr"].Set(pts, Usd.TimeCode(frame))
            if display_color is not None:
                if "usd_color_attr" not in entry:
                    # Lazily create displayColor attribute on first use
                    from pxr import UsdGeom
                    mesh_prim = entry["usd_points_attr"].GetPrim()
                    gprim = UsdGeom.Gprim(mesh_prim)
                    entry["usd_color_attr"] = gprim.GetDisplayColorAttr()
                    if not entry["usd_color_attr"]:
                        entry["usd_color_attr"] = gprim.CreateDisplayColorAttr()
                r, g, b = display_color
                entry["usd_color_attr"].Set(
                    Vt.Vec3fArray([Gf.Vec3f(r, g, b)]), Usd.TimeCode(frame))
        elif self.format == "ply":
            filename = f"{name}_{frame:04d}.ply"
            path = self.abspath / filename
            save_ply(str(path), positions, entry["faces"])

    def finalize(self):
        if self.format == "usd":
            self.usdexporter.finalize()
            print(f"USD animation saved to {self.usdexporter.usd_path}")
        else:
            print(f"PLY frames saved to {self.abspath}")