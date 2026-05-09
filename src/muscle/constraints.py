"""
Constraint building for MuscleSim (pure Python / numpy).

Contains constraint type constants, surface triangle extraction, and the
ConstraintBuilderMixin that provides all create_* methods. Backend-specific
build_constraints lives in muscle.py / muscle_warp.py.
"""
import time

import numpy as np
from scipy.spatial import cKDTree


# Constraint type constants (from pbd_types.h)
PIN           =  157323
ATTACH        =  1650556
TETVOLUME     =  -215389979
TETFIBERNORM  =  -303462111
DISTANCELINE  =  1621136047
TETARAP       =  -92199131
TETARAPNORM   =  -885573303
TETFIBERDGF   =  -403562211
TETSNH        =  -503462311
TETFIBERMILLARD = -503662411

# ARAP flags (from pbd_types.h)
LINEARENERGY = 1 << 0
NORMSTIFFNESS = 1 << 1


class ConstraintBuilderMixin:
    """Mixin providing constraint creation methods (backend-agnostic).

    Requires self to have: pos0_np, tet_np, v_fiber_np, geo, cfg,
    bone_pos, bone_muscle_ids, constraint_configs.
    """

    # -- rest matrix helpers (pure numpy) --

    def _batch_compute_tet_rest_matrices(self):
        """Batch compute rest matrices for all tets.
        Returns cached (restmatrices, volumes, valid) tuple."""
        if hasattr(self, '_cached_tet_rest'):
            return self._cached_tet_rest
        tet_pos = self.pos0_np[self.tet_np]
        cols = tet_pos[:, :3, :] - tet_pos[:, 3:4, :]
        M = np.transpose(cols, (0, 2, 1))
        dets = np.linalg.det(M)
        volumes = dets / 6.0
        valid = np.abs(dets) > 1e-30
        restmatrices = np.zeros_like(M)
        if np.any(valid):
            restmatrices[valid] = np.linalg.inv(M[valid])
        self._cached_tet_rest = (restmatrices, volumes, valid)
        return self._cached_tet_rest

    def compute_tet_rest_matrix(self, pt0, pt1, pt2, pt3, scale=1.0):
        p = self.pos0_np
        M = scale * np.stack([p[pt0] - p[pt3], p[pt1] - p[pt3], p[pt2] - p[pt3]]).T
        detM = np.linalg.det(M)
        if detM == 0:
            return None, 0.0
        return np.linalg.inv(M), detM / 6.0

    def compute_tet_fiber_rest_length(self, pt0, pt1, pt2, pt3):
        restm, volume = self.compute_tet_rest_matrix(pt0, pt1, pt2, pt3, scale=1.0)
        if restm is None:
            return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        materialW = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.v_fiber_np is not None:
            w = self.v_fiber_np[pt0] + self.v_fiber_np[pt1] + self.v_fiber_np[pt2] + self.v_fiber_np[pt3]
            norm = np.linalg.norm(w)
            if norm > 1e-8:
                materialW = w / norm
        materialW = materialW @ restm.T
        return volume, materialW

    def map_pts2tets(self, tet):
        pt2tet = {}
        for i, tet_verts in enumerate(tet):
            for v in tet_verts:
                if v not in pt2tet:
                    pt2tet[v] = []
                pt2tet[v].append(i)
        return pt2tet

    def _normalize_group_names(self, value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            names = [str(value)]
        else:
            names = [str(item) for item in value]
        names = [name for name in names if name]
        return tuple(dict.fromkeys(names)) if names else None

    def _get_bone_vertex_groups(self):
        if hasattr(self, "_cached_bone_vertex_groups"):
            return self._cached_bone_vertex_groups

        group_names = np.full(self.bone_pos.shape[0], "", dtype=object)
        for group_name, indices in getattr(self, "bone_muscle_ids", {}).items():
            group_names[np.asarray(indices, dtype=np.int32)] = str(group_name)

        self._cached_bone_vertex_groups = group_names
        return group_names

    def _resolve_target_indices(self, target_groups):
        if target_groups is None:
            return np.arange(self.bone_pos.shape[0], dtype=np.int32)

        resolved = []
        missing = []
        for group_name in target_groups:
            group_indices = getattr(self, "bone_muscle_ids", {}).get(group_name)
            if group_indices is None:
                missing.append(group_name)
                continue
            resolved.append(np.asarray(group_indices, dtype=np.int32))

        if missing:
            print(f"Warning: bone target groups not found: {missing}")
        if not resolved:
            return np.zeros(0, dtype=np.int32)

        return np.unique(np.concatenate(resolved).astype(np.int32))

    def _filter_sources_by_nearest_group(self, src_indices, src_positions, source_groups):
        if source_groups is None or len(src_indices) == 0:
            return src_indices, src_positions

        bone_tree = cKDTree(self.bone_pos)
        _, nearest_indices = bone_tree.query(src_positions, k=1)
        group_lookup = self._get_bone_vertex_groups()
        keep_mask = np.array(
            [group_lookup[int(idx)] in source_groups for idx in nearest_indices],
            dtype=bool,
        )

        return src_indices[keep_mask], src_positions[keep_mask]

    def _query_bone_targets(self, src_indices, src_positions, params):
        target_groups = self._normalize_group_names(params.get("target_group"))
        source_groups = self._normalize_group_names(params.get("source_nearest_group"))
        if params.get("source_filter_by_nearest_target_group", False) and source_groups is None:
            source_groups = target_groups

        src_indices, src_positions = self._filter_sources_by_nearest_group(
            src_indices,
            src_positions,
            source_groups,
        )
        if len(src_indices) == 0:
            return src_indices, src_positions, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        target_indices = self._resolve_target_indices(target_groups)
        if len(target_indices) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

        target_tree = cKDTree(self.bone_pos[target_indices])
        dists, local_target_indices = target_tree.query(src_positions, k=1)
        tgt_indices = target_indices[np.asarray(local_target_indices, dtype=np.int32)]
        return src_indices, src_positions, np.asarray(dists, dtype=np.float32), np.asarray(tgt_indices, dtype=np.int32)

    # -- constraint creators (return list[dict] with plain lists, no ti/wp) --

    def create_tet_volume_constraint(self, params):
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETVOLUME,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                compressionstiffness=-1.0,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(volumes[i]),
                restvector=[0.0, 0.0, 0.0, 0.0],
                restdir=[0.0, 0.0, 0.0],
            )
            constraints.append(c)
        return constraints

    def _compute_fiber_rest_data(self):
        """Compute per-tet fiber directions transformed to material space.

        Returns (materialW_transformed, volumes, valid, restmatrices) where:
          materialW_transformed: (n_tet, 3) fiber dirs in material coords
          volumes: (n_tet,) tet volumes
          valid: (n_tet,) bool mask for non-degenerate tets
          restmatrices: (n_tet, 3, 3) inverse rest matrices
        """
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        n_tet = len(self.tet_np)
        if self.v_fiber_np is not None:
            fiber_verts = self.v_fiber_np[self.tet_np]
            w = fiber_verts.sum(axis=1)
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            default_w = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))
            norm_ok = (norms > 1e-8).ravel()
            materialW = default_w.copy()
            materialW[norm_ok] = w[norm_ok] / norms[norm_ok]
        else:
            materialW = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))

        materialW_transformed = np.einsum('nj,nkj->nk', materialW, restmatrices)
        return materialW_transformed, volumes, valid, restmatrices

    def create_tet_fiber_constraint(self, params):
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)
        materialW_transformed, volumes, valid, _ = self._compute_fiber_rest_data()

        constraints = []
        for i in range(len(self.tet_np)):
            tet = self.tet_np[i]
            if not valid[i]:
                vol = 0.0
                mw_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vol = float(volumes[i])
                mw_t = materialW_transformed[i]
            c = dict(
                type=TETFIBERNORM,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(vol),
                restvector=[float(mw_t[0]), float(mw_t[1]), float(mw_t[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0
            )
            constraints.append(c)
        return constraints

    def create_tet_fiber_dgf_constraint(self, params):
        """DGF fiber constraint: same geometry as TETFIBERNORM, but uses DGF curves.

        Extra params stored in restdir:
          restdir[0] = sigma0 (peak isometric stress, Pa)
          restdir[1] = contraction_factor (from DGF curve, updated per-step)
          restdir[2] = (reserved)
        """
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)
        sigma0 = params.get('sigma0', 300000.0)
        contraction_factor = params.get('contraction_factor', 0.4)
        materialW_transformed, volumes, valid, _ = self._compute_fiber_rest_data()

        constraints = []
        for i in range(len(self.tet_np)):
            tet = self.tet_np[i]
            if not valid[i]:
                vol = 0.0
                mw_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vol = float(volumes[i])
                mw_t = materialW_transformed[i]
            c = dict(
                type=TETFIBERDGF,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(vol),
                restvector=[float(mw_t[0]), float(mw_t[1]), float(mw_t[2]), 1.0],
                restdir=[float(sigma0), float(contraction_factor), 0.0],
                compressionstiffness=-1.0
            )
            constraints.append(c)
        return constraints

    def create_tet_fiber_millard_constraint(self, params):
        """Millard fiber constraint: same geometry as DGF, but uses Millard spline curves.

        Extra params stored in restdir:
          restdir[0] = sigma0 (peak isometric stress, Pa)
          restdir[1] = contraction_factor (from Millard curve, updated per-step)
          restdir[2] = (reserved)
        """
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)
        sigma0 = params.get('sigma0', 300000.0)
        contraction_factor = params.get('contraction_factor', 0.4)
        materialW_transformed, volumes, valid, _ = self._compute_fiber_rest_data()

        constraints = []
        for i in range(len(self.tet_np)):
            tet = self.tet_np[i]
            if not valid[i]:
                vol = 0.0
                mw_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vol = float(volumes[i])
                mw_t = materialW_transformed[i]
            c = dict(
                type=TETFIBERMILLARD,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(vol),
                restvector=[float(mw_t[0]), float(mw_t[1]), float(mw_t[2]), 1.0],
                restdir=[float(sigma0), float(contraction_factor), 0.0],
                compressionstiffness=-1.0
            )
            constraints.append(c)
        return constraints

    def create_tet_snh_constraint(self, params):
        """Stable Neo-Hookean constraint (coupled deviatoric + volumetric).

        Uses Macklin's XPBD SNH formulation:
          Deviatoric: C_D = ||F||_F - sqrt(3), compliance = 1/(mu*V*dt^2)
          Volumetric: C_H = J - alpha,         compliance = 1/(lam*V*dt^2)
        Solved with a coupled 2x2 linear system per tet.

        params:
            mu: shear modulus (deviatoric stiffness)
            lam: bulk modulus (volumetric stiffness)
            dampingratio: XPBD damping ratio
        """
        mu = params.get('mu', 1000.0)
        lam = params.get('lam', 10000.0)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETSNH,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=mu,
                dampingratio=dampingratio,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(volumes[i]),
                restvector=[0.0, 0.0, 0.0, 1.0],  # rotation quaternion (warm start)
                restdir=[float(lam), 0.0, 0.0],    # restdir[0] = lambda
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_attach_constraints(self, params):
        constraints = []
        target_path = params.get('target_path')
        mask_name = params.get('mask_name')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None
        if mask is None:
            warnings.warn(f"mask '{mask_name}' not found in geometry.")
            return []

        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)
        if len(valid_src_indices) == 0:
            warnings.warn(f"No vertices with mask > {mask_threshold}")
            return []

        src_positions = self.pos0_np[valid_src_indices]
        valid_src_indices, _, dists, tgt_indices = self._query_bone_targets(
            valid_src_indices,
            src_positions,
            params,
        )
        if len(valid_src_indices) == 0:
            print(f"Warning: No valid attach sources remain for mask '{mask_name}' after group filtering.")
            return []

        if not hasattr(self, 'pt2tet'):
            self.pt2tet = self.map_pts2tets(self.tet_np)

        for j, src_idx in enumerate(valid_src_indices):
            tgt_idx = int(tgt_indices[j])
            target_pos = self.bone_pos[tgt_idx]
            tetid = self.pt2tet.get(src_idx, [-1])[0]
            restlength = float(dists[j])
            c = dict(
                type=ATTACH,
                pts=[int(src_idx), -1, int(tgt_idx), -1],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=[0.0, 0.0, 0.0],
                restlength=restlength,
                restvector=[float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_distance_line_constraints(self, params):
        constraints = []
        target_path = params.get('target_path')
        mask_name = params.get('mask_name')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None
        if mask is None:
            warnings.warn(f"mask '{mask_name}' not found in geometry.")
            return []

        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)
        if len(valid_src_indices) == 0:
            warnings.warn(f"No vertices with mask > {mask_threshold}")
            return []

        src_positions = self.pos0_np[valid_src_indices]
        valid_src_indices, src_positions, dists, tgt_indices = self._query_bone_targets(
            valid_src_indices,
            src_positions,
            params,
        )
        if len(valid_src_indices) == 0:
            print(f"Warning: No valid distanceline sources remain for mask '{mask_name}' after group filtering.")
            return []

        target_positions = self.bone_pos[tgt_indices]
        directions = target_positions - src_positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms < 1e-9):
            bad = np.where(norms.ravel() < 1e-9)[0]
            raise ValueError(f"source and target points too close at indices: {valid_src_indices[bad]}")
        directions = directions / norms

        if not hasattr(self, 'pt2tet'):
            self.pt2tet = self.map_pts2tets(self.tet_np)

        for j, src_idx in enumerate(valid_src_indices):
            tgt_idx = int(tgt_indices[j])
            target_pos = target_positions[j]
            direction = directions[j]
            tetid = self.pt2tet.get(src_idx, [-1])[0]
            c = dict(
                type=DISTANCELINE,
                pts=[int(src_idx), -1, int(tgt_idx), -1],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=[0.0, 0.0, 0.0],
                restlength=0.0,
                restvector=[float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0],
                restdir=[float(direction[0]), float(direction[1]), float(direction[2])],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_tet_arap_constraints(self, params):
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETARAP,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(volumes[i]),
                restvector=[0.0, 0.0, 0.0, 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def _collect_raw_constraints(self):
        """Dispatch constraint_configs -> create_* methods, return all_constraints list."""
        self.attach_constraints = []
        self.distanceline_constraints = []

        all_constraints = []
        _t_total = time.perf_counter()
        for params in self.constraint_configs:
            ctype = params['type'].lower()
            new_constraints = []
            if ctype == 'volume':
                new_constraints = self.create_tet_volume_constraint(params)
            elif ctype == 'fiber':
                new_constraints = self.create_tet_fiber_constraint(params)
            elif ctype == 'attach':
                new_constraints = self.create_attach_constraints(params)
                self.attach_constraints.extend(new_constraints)
            elif ctype == 'distanceline':
                new_constraints = self.create_distance_line_constraints(params)
                self.distanceline_constraints.extend(new_constraints)
            elif ctype == 'tetarap':
                new_constraints = self.create_tet_arap_constraints(params)
            elif ctype == 'fiberdgf':
                new_constraints = self.create_tet_fiber_dgf_constraint(params)
            elif ctype == 'fibermillard':
                new_constraints = self.create_tet_fiber_millard_constraint(params)
            elif ctype == 'snh':
                new_constraints = self.create_tet_snh_constraint(params)

            if new_constraints:
                all_constraints.extend(new_constraints)
                print(f"  {params.get('name', ctype)} ({ctype}): {len(new_constraints)} constraints")

        self.raw_constraints = all_constraints.copy()
        _dt_total = time.perf_counter() - _t_total
        return all_constraints, _dt_total


def build_constraint_color_groups(all_constraints):
    """Build graph-colored constraint groups for parallel Gauss-Seidel.

    Two constraints that share any vertex (pts[i] >= 0) are adjacent in the
    dual graph and cannot be in the same color group. Returns a list of
    np.ndarray, each containing constraint indices of one color.
    """
    import warp as wp
    from collections import defaultdict
    from newton._src.sim.graph_coloring import ColoringAlgorithm, color_graph

    n_cons = len(all_constraints)
    if n_cons == 0:
        return []

    # vertex -> constraint id inverted index
    vertex_to_cons = defaultdict(list)
    for cid, c in enumerate(all_constraints):
        for v in c['pts']:
            if v >= 0:
                vertex_to_cons[v].append(cid)

    # build dual graph edges
    edge_set = set()
    for cids in vertex_to_cons.values():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                edge_set.add((min(a, b), max(a, b)))

    if not edge_set:
        return [np.arange(n_cons, dtype=np.int32)]

    edges = np.array(sorted(edge_set), dtype=np.int32)
    edge_wp = wp.array(edges, dtype=int, device="cpu")
    color_groups = color_graph(
        n_cons, edge_wp,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        algorithm=ColoringAlgorithm.MCS,
    )
    return color_groups
