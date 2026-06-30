# tools/osim2mjcf.py
"""Convert the Arm26-subset of an OpenSim `.osim` to a MuJoCo MJCF.

Scope (Arm26): BodySet/Body (mass, mass_center, diagonal inertia), JointSet
WeldJoint + single-axis CustomJoint -> MuJoCo hinge, ForceSet/Thelen2003Muscle
-> spatial tendons through PathPoint sites + MuJoCo <muscle> actuators (Task 3),
WrapCylinder/WrapEllipsoid -> tendon wrap geoms (Task 4).

The parser errors loudly (raises) on any construct it does not handle, so that
larger models (e.g. Gait2354: coordinate-coupler constraints, multi-axis joints,
free root) surface their gaps explicitly instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np


def _vec(text, n=None):
    v = [float(x) for x in (text or "").split()]
    if n is not None and len(v) != n:
        raise ValueError(f"expected {n} floats, got {v!r}")
    return v


def parse_bodies(root):
    bodies = {}
    bset = root.find(".//BodySet/objects")
    for b in bset.findall("Body"):
        meshes = [m.findtext("mesh_file") for m in b.findall(".//Mesh")]
        bodies[b.get("name")] = dict(
            mass=float(b.findtext("mass", "0")),
            com=_vec(b.findtext("mass_center", "0 0 0"), 3),
            inertia=_vec(b.findtext("inertia", "0 0 0 0 0 0"), 6),
            meshes=[m for m in meshes if m],
        )
    return bodies


def _offset_frames(joint):
    frames = {}
    for pof in joint.findall(".//PhysicalOffsetFrame"):
        frames[pof.get("name")] = dict(
            parent=pof.findtext("socket_parent").split("/")[-1],
            translation=_vec(pof.findtext("translation", "0 0 0"), 3),
            orientation=_vec(pof.findtext("orientation", "0 0 0"), 3),
        )
    return frames


def parse_joints(root):
    out = []
    jset = root.find(".//JointSet/objects")
    for j in jset:
        jtype = j.tag
        if jtype not in ("WeldJoint", "CustomJoint"):
            raise NotImplementedError(f"joint type {jtype!r} not handled")
        frames = _offset_frames(j)
        pf = frames[j.findtext("socket_parent_frame")]
        cf = frames[j.findtext("socket_child_frame")]
        if any(abs(x) > 1e-9 for x in cf["translation"]):
            raise NotImplementedError(
                f"non-zero child offset on joint {j.get('name')!r} not handled")
        if any(abs(x) > 1e-9 for x in pf["orientation"] + cf["orientation"]):
            raise NotImplementedError(
                f"non-zero joint frame orientation on {j.get('name')!r} not handled")
        info = dict(type=jtype, name=j.get("name"),
                    parent=pf["parent"], child=cf["parent"],
                    pos=pf["translation"], orient=pf["orientation"])
        if jtype == "CustomJoint":
            axis = coord = None
            st = j.find("SpatialTransform")
            for ta in st.findall("TransformAxis"):
                c = (ta.findtext("coordinates") or "").strip()
                if c and ta.find("LinearFunction") is not None:
                    if axis is not None:
                        raise NotImplementedError(
                            f"multi-DOF CustomJoint {j.get('name')!r} not handled")
                    axis = _vec(ta.findtext("axis"), 3)
                    coord = c
            if axis is None:
                raise NotImplementedError(
                    f"CustomJoint {j.get('name')!r} has no linear rotation axis")
            cobj = j.find(".//Coordinate")
            rng = _vec(cobj.findtext("range"), 2) if cobj is not None else [-3.14159, 3.14159]
            info.update(axis=axis, coord=coord, range=rng)
        out.append(info)
    return out


def parse_muscles(root):
    out = []
    fset = root.find(".//ForceSet/objects")
    if fset is None:
        return out
    for mu in fset.findall("Thelen2003Muscle"):
        name = mu.get("name")
        pts = []
        for k, pp in enumerate(mu.findall(".//PathPoint")):
            body = pp.findtext("socket_parent_frame").split("/")[-1]
            loc = _vec(pp.findtext("location"), 3)
            pts.append((f"{name}_P{k}", body, loc))
        out.append(dict(
            name=name,
            f0=float(mu.findtext("max_isometric_force")),
            ofl=float(mu.findtext("optimal_fiber_length")),
            tsl=float(mu.findtext("tendon_slack_length")),
            pen=float(mu.findtext("pennation_angle_at_optimal", "0") or 0.0),
            points=pts,
        ))
    return out


def _muscle_xml(muscles):
    """Build tendon + actuator XML and populate _SITE_MAP. Returns (tendon, actuator)."""
    global _SITE_MAP
    _SITE_MAP = {}
    tendons = ["  <tendon>\n"]
    acts = ["  <actuator>\n"]
    for mu in muscles:
        tendons.append(f'    <spatial name="{mu["name"]}_t" width="0.002">\n')
        seen_wrap = set()
        # Count how many path points each body has for this muscle; only insert a
        # wrap geom on a body that has >=2 sites in this tendon's path so the geom
        # is guaranteed to be bracketed by two sites (MuJoCo requirement).
        body_site_count: dict[str, int] = {}
        for _sname, body, _loc in mu["points"]:
            body_site_count[body] = body_site_count.get(body, 0) + 1
        body_site_seen: dict[str, int] = {}
        for sname, body, loc in mu["points"]:
            _SITE_MAP.setdefault(body, []).append((sname, loc))
            tendons.append(f'      <site site="{sname}"/>\n')
            body_site_seen[body] = body_site_seen.get(body, 0) + 1
            # Emit at most ONE wrap geom per body after its first site, but only
            # when the body has >=2 sites in this muscle's path so the geom is
            # bracketed by sites (MuJoCo requirement: each geom must have a site
            # directly before AND directly after it; multiple geoms in a row fail).
            if body_site_count.get(body, 0) >= 2 and body_site_seen[body] == 1:
                body_wraps = _WRAP_MAP.get(body, [])
                if body_wraps:
                    gname, _w = body_wraps[0]  # dominant (first) wrap only
                    if gname not in seen_wrap:
                        tendons.append(f'      <geom geom="{gname}"/>\n')
                        seen_wrap.add(gname)
        tendons.append("    </spatial>\n")
        # MuJoCo muscle actuator: force = peak isometric force -> gainprm[2].
        # lengthrange auto-computed by the compiler; if that ever yields a
        # degenerate range, set lengthrange="{tsl+0.5*ofl} {tsl+1.5*ofl}".
        acts.append(
            f'    <muscle name="{mu["name"]}" tendon="{mu["name"]}_t" '
            f'force="{mu["f0"]:.9g}"/>\n')
    tendons.append("  </tendon>\n")
    acts.append("  </actuator>\n")
    return "".join(tendons), "".join(acts)


def parse_wraps(root):
    """Body name -> list of wrap objects fixed to that body."""
    out = {}
    bset = root.find(".//BodySet/objects")
    for b in bset.findall("Body"):
        wl = []
        for w in b.findall(".//WrapObjectSet/objects/*"):
            if (w.findtext("active", "true") or "true").strip() != "true":
                continue
            wl.append(dict(
                name=w.get("name"),
                type=w.tag,
                translation=_vec(w.findtext("translation", "0 0 0"), 3),
                xyz_rot=_vec(w.findtext("xyz_body_rotation", "0 0 0"), 3),
                radius=float(w.findtext("radius", "0") or 0.0),
                length=float(w.findtext("length", "0") or 0.0),
                dims=_vec(w.findtext("dimensions", "0 0 0"), 3) if w.findtext("dimensions") else [0, 0, 0],
                quadrant=(w.findtext("quadrant", "all") or "all").strip(),
            ))
        if wl:
            out[b.get("name")] = wl
    return out


def _euler_xyz_to_quat(rx, ry, rz):
    """Convert body-fixed XYZ Euler angles (radians) to MuJoCo quaternion [w, x, y, z]."""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    # body-fixed x-y-z
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return [w, x, y, z]


def _wrap_xml(root, muscles):
    return ""


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def build_mjcf(osim_path):
    root = ET.parse(osim_path).getroot()
    bodies = parse_bodies(root)
    joints = parse_joints(root)
    muscles = parse_muscles(root)

    # Root bodies hang off ground (WeldJoint parent 'ground' or unparented).
    by_parent = {}
    weld_pos = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
        if j["type"] == "WeldJoint":
            weld_pos[j["child"]] = j["pos"]

    wraps = parse_wraps(root)
    global _WRAP_MAP
    _WRAP_MAP = {}  # body -> list of (geom_name, wrap_dict)
    for body, wl in wraps.items():
        for w in wl:
            gname = f'{w["name"]}_wrap'
            _WRAP_MAP.setdefault(body, []).append((gname, w))

    # Single-wrap-per-body over-approximation: only the first wrap on a body is
    # inserted into each tendon through it. Prefer joint-spanning cylinders over
    # shoulder-cap ellipsoids so the dominant correction fires. For r_humerus
    # this selects TRI_wrap (elbow cylinder) over the TRIlonghh/BIClonghh
    # shoulder spheres, restoring the triceps elbow moment arm. A wrap geom only
    # alters a path it actually penetrates, so this is inert for tendons that do
    # not reach the cylinder. Faithful per-muscle PathWrapSet routing is future
    # work (see spec "Known confounds").
    for _body in _WRAP_MAP:
        _WRAP_MAP[_body].sort(key=lambda gw: gw[1]["type"] != "WrapCylinder")

    def emit(body_name, pos, indent):
        b = bodies[body_name]
        pad = "  " * indent
        s = f'{pad}<body name="{body_name}" pos="{_fmt(pos)}">\n'
        # hinge joint into this body (skip for welded/root bodies)
        hinge = next((j for j in joints
                      if j["type"] == "CustomJoint" and j["child"] == body_name), None)
        if hinge is not None:
            # Arm26 OpenSim coordinates are clamped=false (no hard limits), so
            # emit an unlimited hinge to match the source model. The parsed
            # range stays in the joint dict but is intentionally not emitted.
            s += (f'{pad}  <joint name="{hinge["coord"]}" type="hinge" '
                  f'axis="{_fmt(hinge["axis"])}" limited="false"/>\n')
        if b["mass"] > 0:
            ixx, iyy, izz = b["inertia"][0], b["inertia"][1], b["inertia"][2]
            if any(abs(x) > 1e-12 for x in b["inertia"][3:]):
                raise NotImplementedError(f"off-diagonal inertia on {body_name!r}")
            s += (f'{pad}  <inertial pos="{_fmt(b["com"])}" mass="{b["mass"]:.9g}" '
                  f'diaginertia="{ixx:.9g} {iyy:.9g} {izz:.9g}"/>\n')
        # wrap geoms fixed to this body (Task 4)
        for gname, w in _WRAP_MAP.get(body_name, []):
            quat = _euler_xyz_to_quat(*w["xyz_rot"])
            if w["type"] == "WrapCylinder":
                s += (f'{pad}  <geom name="{gname}" type="cylinder" '
                      f'pos="{_fmt(w["translation"])}" quat="{_fmt(quat)}" '
                      f'size="{w["radius"]:.9g} {max(w["length"], 0.01) / 2:.9g}" '
                      f'contype="0" conaffinity="0" group="3"/>\n')
            elif w["type"] == "WrapEllipsoid":  # sphere of dominant radius (documented approx)
                r = max(w["dims"]) if any(w["dims"]) else max(w["radius"], 0.005)
                s += (f'{pad}  <geom name="{gname}" type="sphere" '
                      f'pos="{_fmt(w["translation"])}" quat="{_fmt(quat)}" '
                      f'size="{r:.9g}" contype="0" conaffinity="0" group="3"/>\n')
            else:
                raise NotImplementedError(f"wrap type {w['type']!r} not handled")
        # sites for muscle path points anchored on this body (Task 3 fills _SITE_MAP)
        for sname, sloc in _SITE_MAP.get(body_name, []):
            s += f'{pad}  <site name="{sname}" pos="{_fmt(sloc)}" size="0.005"/>\n'
        for j in by_parent.get(body_name, []):
            if j["type"] == "CustomJoint":
                s += emit(j["child"], j["pos"], indent + 1)
        s += f"{pad}</body>\n"
        return s

    tendon_xml, actuator_xml = _muscle_xml(muscles)  # fills _SITE_MAP (Task 3); reads _WRAP_MAP (Task 4)

    body_blocks = ""
    for child, pos in weld_pos.items():
        body_blocks += emit(child, pos, 2)

    wrap_assets = _wrap_xml(root, muscles)

    return f"""<mujoco model="arm26">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001"/>
  <worldbody>
{body_blocks}  </worldbody>
{wrap_assets}{tendon_xml}{actuator_xml}</mujoco>
"""


_SITE_MAP = {}
_WRAP_MAP = {}


def main():
    p = argparse.ArgumentParser(description="OpenSim .osim -> MuJoCo MJCF (Arm26 subset)")
    p.add_argument("-i", "--osim", required=True)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    xml = build_mjcf(a.osim)
    with open(a.out, "w") as f:
        f.write(xml)
    print(a.out)


if __name__ == "__main__":
    main()
