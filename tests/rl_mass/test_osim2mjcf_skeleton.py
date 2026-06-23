# tests/rl_mass/test_osim2mjcf_skeleton.py
"""osim2mjcf skeleton: bodies + hinge joints compile in MuJoCo with nq=2."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco
from tools.osim2mjcf import parse_bodies, parse_joints, build_mjcf
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"

def main():
    root = ET.parse(OSIM).getroot()
    bodies = parse_bodies(root)
    assert set(bodies) == {"base", "r_humerus", "r_ulna_radius_hand"}, set(bodies)
    assert abs(bodies["r_humerus"]["mass"] - 1.864572) < 1e-4

    joints = parse_joints(root)
    customs = [j for j in joints if j["type"] == "CustomJoint"]
    assert {j["coord"] for j in customs} == {"r_shoulder_elev", "r_elbow_flex"}
    sh = next(j for j in customs if j["coord"] == "r_shoulder_elev")
    assert sh["parent"] == "base" and sh["child"] == "r_humerus"

    xml = build_mjcf(OSIM)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nq == 2, m.nq
    assert m.njnt == 2, m.njnt
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
    assert names == {"r_shoulder_elev", "r_elbow_flex"}, names
    print("OK")

if __name__ == "__main__":
    main()
