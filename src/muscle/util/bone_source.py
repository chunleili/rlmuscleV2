"""Per-frame bone surface: constant topology + animated vertices.

humanV2 bones arrive either as a directory of Houdini ``bone_<frame>.geo`` files
or as a single animated USD whose Mesh prim(s) carry ``points`` timeSamples.
Both feed the same two consumers — the muscle's mesh-target binding (``mg``) and
the animated collider (``example.humanV2``) — through one interface:

    src.tris             flat int32 triangle indices, AS AUTHORED (no winding flip)
    src.n_verts          vertex count (constant across the sequence)
    src.positions(frame) (N, 3) float32 world-space verts at ``frame``

Winding is left raw: each consumer applies its own policy (the collider flips for
the signed-normal query; the slide binding uses ``tris`` directly).

Pick a backend with ``make_bone_source(cfg, start_frame)``: a top-level
``bone_usd`` key selects the USD backend (optional ``bone_root`` to scope which
prims), otherwise ``bone_dir`` selects the ``.geo`` backend.

The USD backend auto-detects two layouts: a **UsdSkel rig** (SkelRoot) stores
rest points + per-joint animation and is LBS-skinned per frame on demand (the
compact layout — a 200-frame skeleton is a few MB, not hundreds); a **baked**
mesh carries per-frame ``points`` timeSamples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muscle.util.geo_io import GeoFile, read_geo_attr_cached
from muscle.util.util import _check_unzipped


def _triangulate(counts, idx) -> np.ndarray:
    """Fan-triangulate a polygon mesh's ``faceVertexIndices`` into a flat
    ``(M*3,)`` int32 triangle-index list. Already-triangular meshes pass straight
    through. Bone meshes are used only for proximity collision + display, so a
    simple fan (0, k, k+1) over each face is sufficient (walk.usd ships quad/ngon
    bone meshes; otto.usd ships triangles)."""
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if np.all(counts == 3):
        return idx.astype(np.int32)
    tris, o = [], 0
    for c in counts:
        c = int(c)
        for k in range(1, c - 1):              # fan around the first vertex
            tris.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    return np.asarray(tris, dtype=np.int32).reshape(-1)


def _xform_rt(xcache, prim):
    """Local-to-world of *prim* as (R(3,3), T(3,)) in row-vector convention."""
    wm = xcache.GetLocalToWorldTransform(prim)
    R = np.array([[wm[i][j] for j in range(3)] for i in range(3)], dtype=np.float32)
    T = np.array([wm[3][j] for j in range(3)], dtype=np.float32)
    return R, T


def _scaled(arr, scale):
    """Uniform geometry scale (e.g. cm->m asset_scale); no-op at 1.0."""
    if scale == 1.0:
        return arr
    return np.ascontiguousarray((arr * np.float32(scale)).astype(np.float32))


class BoneSource:
    """Interface only — see module docstring."""

    @property
    def tris(self) -> np.ndarray:        # flat int32, as authored
        raise NotImplementedError

    @property
    def n_verts(self) -> int:
        raise NotImplementedError

    def positions(self, frame: int) -> np.ndarray:   # (N,3) float32, world space
        raise NotImplementedError


class GeoBoneSource(BoneSource):
    """Houdini ``bone_<frame>.geo`` directory (the legacy SNU skeleton path).

    Tris live in ``GeoFile.tet_indices`` (Houdini packs surface tris there).
    """

    def __init__(self, bone_dir: str, start_frame: int = 1, scale: float = 1.0):
        self.bone_dir = Path(bone_dir)
        self.start_frame = int(start_frame)
        self._scale = float(scale)
        g0 = GeoFile(self._path(self.start_frame))
        self._tris = np.ascontiguousarray(
            np.asarray(g0.tet_indices, dtype=np.int32).reshape(-1))
        self._n_verts = int(np.asarray(g0.positions).shape[0])

    def _path(self, frame: int) -> str:
        p = str(self.bone_dir / f"bone_{frame}.geo")
        _check_unzipped(p)
        return p

    @property
    def tris(self) -> np.ndarray:
        return self._tris

    @property
    def n_verts(self) -> int:
        return self._n_verts

    def positions(self, frame: int) -> np.ndarray:
        pos = read_geo_attr_cached(self._path(frame), "P")
        out = np.ascontiguousarray(np.asarray(pos, dtype=np.float32))
        return _scaled(out, self._scale)


class UsdBoneSource(BoneSource):
    """Single animated USD: Mesh prim(s) with ``points`` timeSamples.

    Multiple meshes under ``bone_root`` (or the whole stage if ``bone_root`` is
    None) are concatenated into one mesh, indices offset, each prim's points
    pushed to world space by its local-to-world transform (sampled at the rest
    timecode — bone xforms are static; only the points animate).
    """

    def __init__(self, usd_path: str, bone_root: str | None = None,
                 scale: float = 1.0, deroot: bool = False):
        from pxr import Usd, UsdSkel

        self.usd_path = str(usd_path)
        self._scale = float(scale)
        # De-root (skel mode only): subtract the root joint's ground-plane (x,z)
        # world translation each frame so a locomotion clip walks IN PLACE at the
        # origin instead of translating metres away from the static muscle rest.
        # Vertical (y) bob is kept — that is real motion, not drift.
        self._deroot = bool(deroot)
        stage = Usd.Stage.Open(self.usd_path)
        if stage is None:
            raise FileNotFoundError(f"Cannot open bone USD: {usd_path}")
        self._stage = stage

        rest_t = stage.GetStartTimeCode()
        if rest_t == Usd.TimeCode.Default().GetValue():
            rest_t = 1.0
        self._rest_t = float(rest_t)
        self._cache: dict[int, np.ndarray] = {}

        # A UsdSkel rig (SkelRoot present) stores only rest points + per-joint
        # animation; per-frame verts are LBS-skinned on demand (the compact
        # layout). Otherwise points carry per-frame timeSamples (baked layout).
        # When ``bone_root`` is set the SkelRoot is searched ONLY under it — an
        # asset can carry several SkelRoots (walk.usd: Rigid_body_tree / Joint_tree
        # / Muscles) and a bare Traverse() would grab the wrong (often empty) one.
        if bone_root:
            root_prim = stage.GetPrimAtPath(bone_root)
            if not root_prim or not root_prim.IsValid():
                raise ValueError(f"bone_root '{bone_root}' not in {self.usd_path}")
            search = Usd.PrimRange(root_prim)
        else:
            search = stage.Traverse()
        skelroot = next((UsdSkel.Root(p) for p in search
                         if p.IsA(UsdSkel.Root)), None)
        if skelroot is not None:
            self._mode = "skel"
            self._init_skel(skelroot)
        else:
            self._mode = "baked"
            self._init_baked(bone_root)
        print(f"[bone-usd:{self._mode}] {usd_path}: {self._n_verts} verts, "
              f"{self._tris.size // 3} tris, frames "
              f"[{stage.GetStartTimeCode():g}..{stage.GetEndTimeCode():g}]")

    def _init_baked(self, bone_root):
        from pxr import Usd, UsdGeom

        xcache = UsdGeom.XformCache(self._rest_t)
        if bone_root:
            root = self._stage.GetPrimAtPath(bone_root)
            if not root or not root.IsValid():
                raise ValueError(f"bone_root '{bone_root}' not in {self.usd_path}")
            prims = Usd.PrimRange(root)
        else:
            prims = self._stage.Traverse()

        self._prims = []           # list of (UsdGeom.Mesh, R(3,3), T(3,), nverts)
        tri_chunks = []
        offset = 0
        for prim in prims:
            if not prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(prim)
            counts = m.GetFaceVertexCountsAttr().Get()
            idx = m.GetFaceVertexIndicesAttr().Get()
            pts = m.GetPointsAttr().Get(self._rest_t)
            if counts is None or idx is None or pts is None:
                continue
            nv = int(np.asarray(pts).shape[0])
            R, T = _xform_rt(xcache, prim)
            self._prims.append((m, R, T, nv))
            tri_chunks.append(_triangulate(counts, idx) + offset)
            offset += nv

        if not self._prims:
            raise ValueError(
                f"no Mesh prims found in {self.usd_path}"
                + (f" under '{bone_root}'" if bone_root else ""))
        self._tris = np.ascontiguousarray(np.concatenate(tri_chunks))
        self._n_verts = offset

    def _init_skel(self, skelroot):
        from pxr import Usd, UsdGeom, UsdSkel

        cache = UsdSkel.Cache()
        cache.Populate(skelroot, Usd.TraverseInstanceProxies())
        bindings = cache.ComputeSkelBindings(skelroot, Usd.TraverseInstanceProxies())

        self._skin = []   # list of (skelQuery, skinningQuery, rest(N,3), skelPrim, nv)
        tri_chunks = []
        offset = 0
        for b in bindings:
            skel = b.GetSkeleton()
            skel_q = cache.GetSkelQuery(skel)
            for sq in b.GetSkinningTargets():
                prim = sq.GetPrim()
                m = UsdGeom.Mesh(prim)
                counts = m.GetFaceVertexCountsAttr().Get()
                idx = m.GetFaceVertexIndicesAttr().Get()
                rest = m.GetPointsAttr().Get()          # bind-pose points
                if counts is None or idx is None or rest is None:
                    continue
                rest = np.asarray(rest, dtype=np.float32)
                self._skin.append((skel_q, sq, rest, skel.GetPrim(), rest.shape[0]))
                tri_chunks.append(_triangulate(counts, idx) + offset)
                offset += rest.shape[0]

        if not self._skin:
            raise ValueError(f"no skinned Mesh under SkelRoot in {self.usd_path}")
        self._tris = np.ascontiguousarray(np.concatenate(tri_chunks))
        self._n_verts = offset

    @property
    def tris(self) -> np.ndarray:
        return self._tris

    @property
    def n_verts(self) -> int:
        return self._n_verts

    def positions(self, frame) -> np.ndarray:
        # `frame` may be fractional (sub-frame bone sampling): UsdSkel/baked points
        # interpolate natively at any TimeCode. Only INTEGER frames are cached --
        # fractional sub-poses are unique per step and would blow RAM (e.g. 10x225
        # poses x 36k verts ~ 1 GB).
        cache = float(frame).is_integer()
        if cache:
            cached = self._cache.get(frame)
            if cached is not None:
                return cached
        out = (self._skinned(frame) if self._mode == "skel"
               else self._baked(frame))
        out = _scaled(out, self._scale)
        if cache:
            self._cache[frame] = out
        return out

    def _baked(self, frame: int) -> np.ndarray:
        chunks = []
        for m, R, T, nv in self._prims:
            pts = m.GetPointsAttr().Get(float(frame))
            if pts is None:                       # static prim: fall back to rest
                pts = m.GetPointsAttr().Get(self._rest_t)
            pts = np.asarray(pts, dtype=np.float32)
            chunks.append(pts @ R + T)
        return np.ascontiguousarray(np.vstack(chunks).astype(np.float32))

    def _skinned(self, frame: int) -> np.ndarray:
        from pxr import Usd, UsdGeom, Vt

        tc = Usd.TimeCode(float(frame))
        xcache = UsdGeom.XformCache(tc)
        chunks = []
        for skel_q, sq, rest, skel_prim, nv in self._skin:
            # LBS: rest (geom-bind space) -> skeleton space at this frame ...
            xforms = skel_q.ComputeSkinningTransforms(tc)
            pts = Vt.Vec3fArray.FromNumpy(rest.copy())
            sq.ComputeSkinnedPoints(xforms, pts, tc)
            arr = np.asarray(pts, dtype=np.float32)
            # ... then skeleton space -> world.
            R, T = _xform_rt(xcache, skel_prim)
            arr = arr @ R + T
            if self._deroot:                          # pelvis (joint 0) -> origin in x,z
                skelT = skel_q.ComputeJointSkelTransforms(tc)
                root_w = np.asarray(skelT[0].ExtractTranslation(), np.float32) @ R + T
                arr[:, 0] -= root_w[0]
                arr[:, 2] -= root_w[2]
            chunks.append(arr)
        return np.ascontiguousarray(np.vstack(chunks).astype(np.float32))


def make_bone_source(cfg: dict, start_frame: int = 1) -> BoneSource:
    """Build the bone source from config: ``bone_usd`` -> USD, else ``bone_dir``.

    ``asset_scale`` uniformly scales geometry on load (e.g. 0.01 for a cm asset
    into the solver's metre world); muscle (UsdTetRest) must use the same value.
    """
    scale = float(cfg.get("asset_scale", 1.0))
    usd = cfg.get("bone_usd")
    if usd:
        return UsdBoneSource(usd, cfg.get("bone_root"), scale=scale,
                             deroot=bool(cfg.get("bone_deroot", False)))
    bone_dir = cfg.get("bone_dir")
    if not bone_dir:
        raise ValueError("config needs 'bone_usd' or 'bone_dir' for bones")
    return GeoBoneSource(bone_dir, start_frame, scale=scale)
