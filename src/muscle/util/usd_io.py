from __future__ import annotations

"""Minimal USD read/write facade for RLMuscle.

Public API:
- `UsdIO`: one object for read + layered edit operations.
- `usd_args`: CLI argument helper for teaching examples.
"""

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

import numpy as np

from muscle.util.mesh_io import _Y_TO_Z, build_surface_tris
from muscle.util.mesh_utils import fix_tet_winding

ColorRgb = tuple[float, float, float]


@dataclass
class UsdMesh:
    mesh_path: str
    vertices: np.ndarray          # (N, 3) float32
    faces: np.ndarray             # (T, 3) int32
    color: ColorRgb = (0.7, 0.7, 0.7)
    tets: np.ndarray | None = None          # (M, 4) int32
    primvars: dict | None = None            # {"materialW": ndarray, ...}


@dataclass
class WarpMeshData:
    """Per-mesh warp arrays for viewer.log_mesh() rendering."""
    name: str               # prim path
    rest_pos: "wp.array"    # rest-pose vertices (vec3, immutable)
    pos: "wp.array"         # deformed vertices  (vec3, written each frame)
    tri_indices: "wp.array" # triangle indices   (int32)


def usd_args(
    usd_path: str = "data/muscle/model/bicep.usd",
    output_path: str = "output.usd",
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add shared USD-layering CLI arguments for teaching examples."""
    if parser is None:
        import newton.examples
        parser = newton.examples.create_parser()
    parser.set_defaults(output_path=output_path)
    parser.add_argument("--usd-path", type=str, default=usd_path,
                        help="data/muscle/model/bicep.usd")
    parser.add_argument("--usd-root-path", type=str, default="/",
                        help="USD prim path to load.")
    parser.add_argument("--use_layered_usd", "--use-layered-usd",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Write runtime edits into a layered USD file.")
    parser.add_argument("--copy_usd", "--copy-usd",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Copy source USD to output directory before writing layered edits.")
    return parser


# ---------------------------------------------------------------------------
# Geometry helpers (pure math, no USD dependency)
# ---------------------------------------------------------------------------

def _triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygonal faces into (T, 3) int32 triangle array."""
    tris: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts.tolist():
        count = int(count)
        face = face_indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        a = int(face[0])
        for i in range(1, count - 1):
            tris.append((a, int(face[i]), int(face[i + 1])))
    return np.asarray(tris, dtype=np.int32)



def _read_display_color(prim) -> ColorRgb:
    """Read displayColor primvar from a prim, default gray."""
    from pxr import UsdGeom
    try:
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        if not pv:
            return (0.7, 0.7, 0.7)
        raw = pv.Get()
        if raw is None:
            return (0.7, 0.7, 0.7)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size < 3:
            return (0.7, 0.7, 0.7)
        rgb = np.clip(arr.reshape(-1, 3).mean(axis=0) if arr.ndim > 1 else arr[:3], 0.0, 1.0)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    except Exception:
        return (0.7, 0.7, 0.7)


# ---------------------------------------------------------------------------
# UsdIO – single public class for read + layered edit
# ---------------------------------------------------------------------------

class UsdIO:
    """Minimal USD API for mesh reading and layered editing.

    Quick start::

        usd = UsdIO("data/muscle/model/bicep.usd").read()
        for mesh in usd.meshes:
            print(mesh.mesh_path, mesh.vertices.shape)

        with usd.start("output/bicep.anim.usda"):
            usd.set_runtime("activation", 0.5, frame=0)
    """

    def __init__(self, source_usd_path: str, *, root_path: str = "/",
                 y_up_to_z_up: bool = False):
        self.source_usd_path = os.path.abspath(str(source_usd_path))
        self.root_path = root_path.strip() or "/"
        self.y_up_to_z_up = bool(y_up_to_z_up)
        self._meshes: list[UsdMesh] | None = None
        # Layered editor state (initialized by start())
        self._stage = None
        self._layer = None
        self._frame_attr = None
        self._time_start: int | None = None
        self._time_end: int | None = None
        self._output_path: str | None = None
        self._runtime_prim = "/anim/runtime"
        # pxr modules (lazy, set by start())
        self._Gf = None
        self._Sdf = None
        self._Vt = None
        self._has_read = False

        if not os.path.isfile(self.source_usd_path):
            raise FileNotFoundError(f"USD source file not found: {self.source_usd_path}")
        
        if not self._has_read:
            self.read()
            

    # -------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------

    def read(self) -> "UsdIO":
        if self._has_read:
            return self
        if self._meshes is not None:
            return self
        from pxr import Gf, Usd, UsdGeom

        root_path = self.root_path
        stage = Usd.Stage.Open(self.source_usd_path)
        if stage is None:
            raise FileNotFoundError(f"Failed to open USD file: {self.source_usd_path}")
        self._stage = stage

        apply_rotation = self.y_up_to_z_up and str(UsdGeom.GetStageUpAxis(stage)).upper() == "Y"
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        if root_path == "/":
            prim_iter = stage.Traverse()
        else:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                raise ValueError(f"USD root path does not exist: {root_path}")
            prim_iter = Usd.PrimRange(root_prim)

        meshes: list[UsdMesh] = []
        for prim in prim_iter:
            is_mesh = prim.IsA(UsdGeom.Mesh)
            is_tet = prim.IsA(UsdGeom.TetMesh)
            if not is_mesh and not is_tet:
                continue

            tets = None
            if is_mesh:
                verts, tris = None, None
                try:
                    import newton.usd as nusd
                    nm = nusd.get_mesh(prim)
                    verts = np.asarray(nm.vertices, dtype=np.float32)
                    idx = np.asarray(nm.indices, dtype=np.int32)
                    tris = idx.reshape(-1, 3) if idx.ndim == 1 else idx
                except Exception:
                    pass
                if verts is None:
                    m = UsdGeom.Mesh(prim)
                    pts = m.GetPointsAttr().Get()
                    cnts = m.GetFaceVertexCountsAttr().Get()
                    idx = m.GetFaceVertexIndicesAttr().Get()
                    if pts is None or cnts is None or idx is None:
                        continue
                    verts = np.asarray(pts, dtype=np.float32)
                    tris = _triangulate_faces(np.asarray(cnts, dtype=np.int32),
                                              np.asarray(idx, dtype=np.int32))
                    if tris.size == 0:
                        continue

            elif is_tet:
                tm = UsdGeom.TetMesh(prim)
                pts_val = tm.GetPointsAttr().Get()
                if pts_val is None:
                    continue
                verts = np.asarray(pts_val, dtype=np.float32)

                tris = None
                surf_attr = tm.GetSurfaceFaceVertexIndicesAttr()
                if surf_attr and surf_attr.IsValid():
                    sv = surf_attr.Get()
                    if sv is not None:
                        raw = np.asarray(sv, dtype=np.int32)
                        if raw.size > 0:
                            tris = raw.reshape(-1, 3) if raw.ndim == 1 else raw

                tet_attr = tm.GetTetVertexIndicesAttr()
                if tet_attr and tet_attr.IsValid():
                    tv = tet_attr.Get()
                    if tv is not None:
                        raw = np.asarray(tv, dtype=np.int32)
                        if raw.size > 0:
                            tets = raw.reshape(-1, 4) if raw.ndim == 1 else raw

                if (tris is None or tris.size == 0) and tets is not None and tets.shape[0] > 0:
                    valid = np.all((tets >= 0) & (tets < verts.shape[0]), axis=1)
                    tets = fix_tet_winding(verts, tets[valid].copy())
                    tris = build_surface_tris(tets, verts)

                if tris is None or tris.size == 0:
                    continue

            if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
                continue

            world = xform_cache.GetLocalToWorldTransform(prim)
            verts = np.asarray(
                [world.Transform(Gf.Vec3d(*p.astype(float))) for p in verts],
                dtype=np.float32,
            )
            if apply_rotation:
                verts = (verts @ _Y_TO_Z.T).astype(np.float32)
            if tets is not None and tets.size > 0:
                tets = fix_tet_winding(verts, tets.copy())

            # Read primvars (skip display-only ones)
            pv_api = UsdGeom.PrimvarsAPI(prim)
            primvars = {}
            for pv in pv_api.GetPrimvars():
                name = pv.GetPrimvarName()
                if name in ("displayColor", "displayOpacity"):
                    continue
                val = pv.Get()
                if val is not None:
                    primvars[name] = np.asarray(val)

            meshes.append(UsdMesh(
                mesh_path=str(prim.GetPath()),
                vertices=verts,
                faces=np.asarray(tris, dtype=np.int32),
                color=_read_display_color(prim),
                tets=tets,
                primvars=primvars if primvars else None,
            ))

        if not meshes:
            raise ValueError(f"No renderable Mesh/TetMesh found under '{root_path}' in: {self.source_usd_path}")
        self._meshes = meshes
        self._has_read = True
        return self

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def meshes(self) -> list[UsdMesh]:
        if self._meshes is None:
            raise RuntimeError("Call read() first.")
        return self._meshes

    @property
    def focus_points(self) -> np.ndarray:
        m = self.meshes
        return np.vstack([x.vertices for x in m]).astype(np.float32) if m else np.zeros((0, 3), np.float32)

    @property
    def output_path(self) -> str | None:
        return self._output_path

    def find_mesh(self, keyword: str) -> UsdMesh | None:
        """Find first mesh whose prim_path contains *keyword*."""
        for m in self.meshes:
            if keyword in m.mesh_path:
                return m
        return None

    def find_meshes(self, keyword: str) -> list[UsdMesh]:
        """Find all meshes whose prim_path contains *keyword*."""
        return [m for m in self.meshes if keyword in m.mesh_path]

    # -------------------------------------------------------------------
    # Warp integration
    # -------------------------------------------------------------------

    def warp_mesh_data(self) -> list[WarpMeshData]:
        """Return per-mesh warp arrays ready for viewer.log_mesh() rendering."""
        import warp as wp
        result = []
        for mesh in self.meshes:
            verts_np = np.asarray(mesh.vertices, dtype=np.float32)
            faces_np = np.asarray(mesh.faces.reshape(-1), dtype=np.int32)
            result.append(WarpMeshData(
                name=mesh.mesh_path,
                rest_pos=wp.array(verts_np, dtype=wp.vec3),
                pos=wp.array(verts_np, dtype=wp.vec3),
                tri_indices=wp.array(faces_np, dtype=wp.int32),
            ))
        return result


    # -------------------------------------------------------------------
    # Layered USD editing
    # -------------------------------------------------------------------

    def start(self, output_path: str | None = None, *,
              copy_usd: bool = True, runtime_prim: str = "/anim/runtime") -> "UsdIO":
        """Open a layered USD stage for writing runtime edits."""
        from pxr import Gf, Sdf, Usd, Vt
        self._Gf, self._Sdf, self._Vt = Gf, Sdf, Vt

        if output_path is None:
            stem = os.path.splitext(os.path.basename(self.source_usd_path))[0]
            output_path = os.path.abspath(os.path.join("output", f"{stem}.anim.usd"))
        self.close()

        self._output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        if copy_usd:
            self._copy_source()

        # Always start fresh: clear existing layer or create new one.
        layer = Sdf.Layer.FindOrOpen(self._output_path)
        if layer:
            layer.Clear()
        else:
            layer = Sdf.Layer.CreateNew(self._output_path)
        src_rel = os.path.relpath(self.source_usd_path, os.path.dirname(self._output_path)).replace("\\", "/")
        if src_rel not in list(layer.subLayerPaths):
            layer.subLayerPaths = list(layer.subLayerPaths) + [src_rel]
            layer.Save()
        self._stage = Usd.Stage.Open(layer)
        self._stage.SetEditTarget(layer)
        self._layer = layer
        self._time_start = None
        self._time_end = None

        # Metadata scope
        dp = self._stage.GetDefaultPrim()
        scope_path = f"{dp.GetPath()}/anim" if dp and dp.IsValid() else "/anim"
        scope = self._stage.DefinePrim(scope_path, "Scope")
        self._frame_attr = scope.CreateAttribute("frameNum", Sdf.ValueTypeNames.Int, custom=True)
        scope.CreateAttribute("sourceUsd", Sdf.ValueTypeNames.String, custom=True).Set(
            self.source_usd_path.replace("\\", "/"))
        scope.CreateAttribute("copyUsd", Sdf.ValueTypeNames.Bool, custom=True).Set(copy_usd)

        self._runtime_prim = runtime_prim.strip() or "/anim/runtime"
        self._stage.DefinePrim(self._runtime_prim, "Scope")
        return self

    def set_runtime(self, name: str, value, *,
                    value_type: str | None = None, frame: int | None = None) -> bool:
        """Write a scalar value under the runtime scope prim."""
        return self.set_custom(self._runtime_prim, name, value, value_type=value_type, frame=frame)

    def set_custom(self, prim_path: str, name: str, value, *,
                   value_type: str | None = None, frame: int | None = None) -> bool:
        """Write a custom scalar property on a prim (auto-creates Scope if missing)."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self._stage.DefinePrim(prim_path, "Scope")
            prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        sdf_t = self._sdf_type(value, value_type)
        attr = prim.CreateAttribute(name, sdf_t, custom=True)
        v = self._coerce(value, sdf_t)
        if frame is None:
            attr.Set(v)
        else:
            attr.Set(v, int(frame))
            self._mark_frame(int(frame))
        return True

    def set_points(self, prim_path: str, points, *, frame: int | None = None) -> bool:
        """Write per-vertex positions to a Mesh prim's points attribute.

        Automatically converts Z-up -> Y-up if y_up_to_z_up was set on read.
        """
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        pts_np = points.numpy() if hasattr(points, 'numpy') else np.asarray(points, dtype=np.float32)
        if self.y_up_to_z_up:
            pts_np = pts_np @ _Y_TO_Z  # Z-up -> Y-up (result stays float32)
        vt_points = self._Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts_np, dtype=np.float32))
        attr = prim.GetAttribute("points")
        if frame is None:
            attr.Set(vt_points)
        else:
            attr.Set(vt_points, int(frame))
            self._mark_frame(int(frame))
        return True

    def set_xform(self, prim_path: str, pos, quat_wxyz, *,
                  frame: int | None = None) -> bool:
        """Write a rigid-body transform (translate + orient) to a prim.

        Args:
            pos: Translation array-like [x, y, z].
            quat_wxyz: Quaternion [w, x, y, z] (USD convention).
        """
        from pxr import UsdGeom
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False

        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        if not ops:
            xf.AddTranslateOp()
            xf.AddOrientOp()
            ops = xf.GetOrderedXformOps()

        p = np.asarray(pos, dtype=np.float64).ravel()
        q = np.asarray(quat_wxyz, dtype=np.float64).ravel()

        if self.y_up_to_z_up:
            p = (p.astype(np.float32) @ _Y_TO_Z).astype(np.float64)

        t = int(frame) if frame is not None else self._Sdf.TimeCode.Default()
        if len(ops) >= 1:
            ops[0].Set(self._Gf.Vec3d(p[0], p[1], p[2]), t)
        if len(ops) >= 2:
            ops[1].Set(self._Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])), t)
        if frame is not None:
            self._mark_frame(int(frame))
        return True

    def save(self):
        if self._stage:
            for layer in (self._stage.GetRootLayer(), self._layer):
                if layer:
                    layer.Save()

    def close(self):
        if self._stage:
            self.save()
            self._stage = None
            self._layer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _copy_source(self):
        import shutil
        import tempfile
        from pathlib import Path

        src = Path(self.source_usd_path).resolve()
        out_path = Path(self._output_path).resolve()
        dst = out_path.parent / f"{out_path.stem}.source{src.suffix}"

        if dst == src:
            self.source_usd_path = str(dst)
            return

        # Copy through a temporary file first so Windows never sees a partial overwrite.
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=str(dst.parent),
            prefix=f"{dst.stem}.",
            suffix=dst.suffix,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            shutil.copy2(src, tmp_path)
            os.replace(tmp_path, dst)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        self.source_usd_path = str(dst)

    def _touch_time(self, frame: int):
        if self._time_start is None or frame < self._time_start:
            self._time_start = frame
        if self._time_end is None or frame > self._time_end:
            self._time_end = frame
        for layer in (self._stage.GetRootLayer(), self._layer):
            if layer:
                layer.startTimeCode = float(self._time_start)
                layer.endTimeCode = float(self._time_end)

    def _mark_frame(self, frame: int):
        self._frame_attr.Set(frame, frame)
        self._touch_time(frame)

    def _sdf_type(self, value, value_type: str | None):
        """Resolve Sdf type: explicit name or infer from Python type."""
        Sdf = self._Sdf
        if value_type:
            return getattr(Sdf.ValueTypeNames, value_type)
        if isinstance(value, bool):
            return Sdf.ValueTypeNames.Bool
        if isinstance(value, int):
            return Sdf.ValueTypeNames.Int
        if isinstance(value, float):
            return Sdf.ValueTypeNames.Float
        if isinstance(value, str):
            return Sdf.ValueTypeNames.String
        raise ValueError(f"Cannot infer USD type from {type(value).__name__}. Pass value_type=.")

    def _coerce(self, value, sdf_type):
        """Coerce Python value to USD-compatible type."""
        tn = str(sdf_type)
        if "Float3" in tn or "Color3f" in tn:
            a = np.asarray(value, dtype=np.float32).ravel()
            return self._Gf.Vec3f(float(a[0]), float(a[1]), float(a[2]))
        if "Bool" in tn:
            return bool(value)
        if "Int" in tn:
            return int(value)
        if "Float" in tn or "Double" in tn:
            return float(value)
        if "String" in tn:
            return str(value)
        return value


def build_bone_prim_map(usd: UsdIO, sim) -> dict[str, np.ndarray]:
    """Map each bone USD prim path to vertex indices in sim.bone_pos_field.

    Matches prim path keywords (e.g. ``/character/bone/L_radius/...``) against
    ``sim.bone_muscle_ids`` group names (e.g. ``"L_radius"``).
    """
    mapping: dict[str, np.ndarray] = {}
    for bm in usd.bone_meshes:
        for group_name, indices in sim.bone_muscle_ids.items():
            if group_name.lower() in bm.mesh_path.lower():
                mapping[bm.mesh_path] = np.asarray(indices, dtype=np.int32)
                break
    return mapping


__all__ = ["UsdIO", "usd_args", "UsdMesh", "WarpMeshData", "build_bone_prim_map"]
