"""Minimal mesh utilities for muscle simulation.

Only the helpers needed by the warp coupling path are kept. The original
RLMuscle ``mesh_utils.py`` also contained Newton VBD-vmuscle integration
(``add_soft_vmuscle_mesh``, ``create_cylinder_tet_mesh`` etc.) which the
XPBD-warp pipeline does not use.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def fix_tet_winding(vertices, tets):
    """Fix tet winding so all tetrahedra have positive volume.

    Swaps vertex 0 and 1 for any inverted tet (negative signed volume).
    """
    tets = tets.copy()
    p0 = vertices[tets[:, 0]]
    p1 = vertices[tets[:, 1]]
    p2 = vertices[tets[:, 2]]
    p3 = vertices[tets[:, 3]]
    signed6v = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
    inverted = signed6v < 0.0
    if np.any(inverted):
        t0 = tets[inverted, 0].copy()
        tets[inverted, 0] = tets[inverted, 1]
        tets[inverted, 1] = t0
    return tets


class MeshDistortionError(RuntimeError):
    """Raised when mesh quality check detects severe element distortion."""
    pass


def check_mesh_quality(positions, tet_indices, tet_poses,
                       detF_min=0.05, detF_max=50.0, step=None):
    """Check mesh quality by computing det(F) for all tetrahedra.

    Raises ``MeshDistortionError`` when any tet is inverted (det(F)<=0) or
    more than 10% of tets are outside [detF_min, detF_max].
    """
    n = len(tet_indices)
    detF = np.empty(n, dtype=np.float64)

    for e in range(n):
        i, j, k, l = tet_indices[e]
        Ds = np.column_stack([
            positions[j] - positions[i],
            positions[k] - positions[i],
            positions[l] - positions[i],
        ])
        F = Ds @ tet_poses[e]
        detF[e] = np.linalg.det(F)

    step_tag = f"step={step} " if step is not None else ""

    n_inverted = int(np.sum(detF <= 0.0))
    if n_inverted > 0:
        worst = float(detF.min())
        worst_id = int(np.argmin(detF))
        msg = (f"{step_tag}Mesh quality CRITICAL: {n_inverted}/{n} tets inverted "
               f"(det(F)<=0). Worst tet {worst_id}: det(F)={worst:.6f}")
        logger.error(msg)
        raise MeshDistortionError(msg)

    n_bad = int(np.sum((detF < detF_min) | (detF > detF_max)))
    bad_frac = n_bad / n

    if n_bad > 0:
        worst_min = float(detF.min())
        worst_max = float(detF.max())
        logger.warning(
            f"{step_tag}Mesh quality: {n_bad}/{n} tets outside "
            f"[{detF_min}, {detF_max}] range. "
            f"det(F) in [{worst_min:.4f}, {worst_max:.4f}]"
        )

    if bad_frac > 0.1:
        msg = (f"{step_tag}Mesh quality CRITICAL: {n_bad}/{n} ({bad_frac:.0%}) tets "
               f"severely distorted (det(F) outside [{detF_min}, {detF_max}]). "
               f"det(F) in [{float(detF.min()):.4f}, {float(detF.max()):.4f}]")
        logger.error(msg)
        raise MeshDistortionError(msg)

    if step is not None and step % 100 == 0:
        logger.info(
            f"{step_tag}Mesh quality OK: det(F) in "
            f"[{float(detF.min()):.4f}, {float(detF.max()):.4f}]"
        )

    return detF
