"""Export per-joint motion from Houdini after FBIK retargeting.

Run this from a Python Shell, Python Source, or Python ROP in Houdini.
Reads point attributes from the SOP at SOP_PATH:
    idx            (int)            -- joint index (== row in output arrays)
    name           (string)         -- joint name (e.g. R_upperarm)
    transform      (matrix4 / m3)   -- world transform, animated
    rest_transform (matrix4 / m3)   -- world transform at the rest frame

Writes JSON to OUT_PATH with shape:
{
  "fps": 30,
  "frame_start": 0,
  "frame_end": 300,
  "joint_names":     [n0, n1, ...],                    # length J, index by idx
  "rest_transforms": [[16 floats row-major], ...],     # length J
  "frames": [                                           # length F
      {"frame": 0, "transforms": [[16 floats], ...]},  # length J each
      ...
  ]
}

Conventions:
  - matrices are row-major 4x4, Houdini default (row-vector, translation in row 3)
  - if `transform` is matrix3 (9 floats), the script assembles a 4x4 with
    point P as translation row.
"""

import json
import hou


# --------- USER: edit these ---------
SOP_PATH    = "/obj/geo1/OUT_ANIM_JOINTS"            # SOP downstream of FBIK
OUT_PATH    = "otto_motion.json"
FRAME_START = int(hou.playbar.frameRange()[0])
FRAME_END   = int(hou.playbar.frameRange()[1])
REST_FRAME  = FRAME_START                    # frame to sample rest_transform
# ------------------------------------


def _to_4x4(val, P):
    """Coerce a transform attrib value into a flat list of 16 row-major floats.

    Accepts:
      - hou.Matrix4 / 16-tuple                -> as-is
      - hou.Matrix3 / 9-tuple                 -> embed in 4x4, use P as translation
      - tuple of (m3_or_m4) already flattened
    """
    flat = list(val)
    if len(flat) == 16:
        return [float(x) for x in flat]
    if len(flat) == 9:
        m3 = flat
        out = [
            m3[0], m3[1], m3[2], 0.0,
            m3[3], m3[4], m3[5], 0.0,
            m3[6], m3[7], m3[8], 0.0,
            float(P[0]), float(P[1]), float(P[2]), 1.0,
        ]
        return out
    raise ValueError(f"Unsupported transform length {len(flat)} (need 9 or 16)")


def _read_geo(sop_path):
    node = hou.node(sop_path)
    if node is None:
        raise RuntimeError(f"SOP not found: {sop_path}")
    geo = node.geometry()
    if geo is None:
        raise RuntimeError(f"No geometry on {sop_path}; cook the node first")
    return geo


def _read_joints_at_current_frame(sop_path, want_rest=False):
    """Return (names_by_idx, transforms_by_idx, rest_by_idx)."""
    geo = _read_geo(sop_path)
    pts = geo.points()
    n = len(pts)

    names_by_idx = [None] * n
    xforms_by_idx = [None] * n
    rest_by_idx = [None] * n if want_rest else None

    has_idx = geo.findPointAttrib("idx") is not None
    for i, pt in enumerate(pts):
        idx = int(pt.attribValue("idx")) if has_idx else i
        if idx < 0 or idx >= n:
            raise RuntimeError(f"point {i}: idx={idx} out of range [0,{n})")
        P = pt.position()
        names_by_idx[idx] = str(pt.attribValue("name"))
        xforms_by_idx[idx] = _to_4x4(pt.attribValue("transform"), P)
        if want_rest:
            rest_by_idx[idx] = _to_4x4(pt.attribValue("rest_transform"), P)
    return names_by_idx, xforms_by_idx, rest_by_idx


def main():
    out = {
        "fps": float(hou.fps()),
        "frame_start": FRAME_START,
        "frame_end": FRAME_END,
        "joint_names": None,
        "rest_transforms": None,
        "frames": [],
    }

    hou.setFrame(REST_FRAME)
    names, _, rest = _read_joints_at_current_frame(SOP_PATH, want_rest=True)
    out["joint_names"] = names
    out["rest_transforms"] = rest
    print(f"[export_otto_motion] {len(names)} joints, rest frame {REST_FRAME}")

    for f in range(FRAME_START, FRAME_END + 1):
        hou.setFrame(f)
        names_f, xforms_f, _ = _read_joints_at_current_frame(SOP_PATH, want_rest=False)
        if names_f != names:
            raise RuntimeError(f"joint name order changed at frame {f}")
        out["frames"].append({"frame": f, "transforms": xforms_f})
        if f % 30 == 0 or f == FRAME_START or f == FRAME_END:
            print(f"  frame {f}/{FRAME_END}")

    out_path = hou.expandString(OUT_PATH)
    with open(out_path, "w") as fp:
        json.dump(out, fp)
    print(f"[export_otto_motion] wrote {out_path}  "
          f"({len(out['frames'])} frames, {len(names)} joints)")

main()
