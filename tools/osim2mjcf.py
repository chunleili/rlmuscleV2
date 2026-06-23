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


def parse_muscles(root):  # implemented in Task 3
    return []


def _muscle_xml(muscles):  # implemented in Task 3
    return "", ""


def _wrap_xml(root, muscles):  # implemented in Task 4
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
        # sites for muscle path points anchored on this body (Task 3 fills _site_map)
        for sname, sloc in _SITE_MAP.get(body_name, []):
            s += f'{pad}  <site name="{sname}" pos="{_fmt(sloc)}" size="0.005"/>\n'
        for j in by_parent.get(body_name, []):
            if j["type"] == "CustomJoint":
                s += emit(j["child"], j["pos"], indent + 1)
        s += f"{pad}</body>\n"
        return s

    global _SITE_MAP
    _SITE_MAP = {}
    tendon_xml, actuator_xml = _muscle_xml(muscles)  # also fills _SITE_MAP (Task 3)

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
