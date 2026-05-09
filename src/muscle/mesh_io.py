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


def _load_mesh_usd(path: Path, y_up_to_z_up: bool = False):
    """Load muscle TetMesh from USD.

    Returns (positions, tets, fibers, tendon_mask, geo_namespace).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {path}")

    tet_prim = None
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


def _load_bone_usd(usd_path: Path, bone_root: str = "/character/bone"):
    """Load bone mesh from USD (multiple Mesh prims under *bone_root*).

    Returns (None, positions, indices, muscle_ids).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")

    root = stage.GetPrimAtPath(bone_root)
    if not root or not root.IsValid():
        return None, np.zeros((0, 3), np.float32), np.zeros(0, np.int32), {}

    all_pos, all_idx, all_mid = [], [], []
    offset = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts_raw = m.GetPointsAttr().Get()
        if pts_raw is None:
            continue
        pts = np.asarray(pts_raw, dtype=np.float32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

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

def load_mesh(path: Path):
    """Load muscle TetMesh from USD.

    Returns (positions, tets, fibers, tendon_mask, geo).
    """
    if path is None:
        raise ValueError("No mesh path provided.")
    return _load_mesh_usd(path, y_up_to_z_up=False)


def load_bone_mesh(path: Path):
    """Load bone surface mesh from USD.

    Returns (bone_geo, bone_pos, bone_indices, bone_muscle_ids) where:
    - bone_geo: always None (legacy slot, kept for tuple stability)
    - bone_pos: np.ndarray shape (N, 3)
    - bone_indices: np.ndarray flat int32
    - bone_muscle_ids: dict mapping muscle_id -> vertex index array, derived
      from prim hierarchy under /character/bone/<bone_name>/Mesh.
    """
    return _load_bone_usd(path)


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
        os.makedirs(path, exist_ok=True)
        self.format = format
        self.abspath = Path(path).absolute()
        self._extra_meshes = {}  # name -> {faces, usd_points_attr (usd only)}
        if format == "usd":
            usd_path = str(self.abspath) + ".usd"
            self.usdexporter = UsdTetExporter(
                tet_indices=tet_indices, usd_path=usd_path,
                prim_path=prim_path, fps=fps)
        elif format == "ply":
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