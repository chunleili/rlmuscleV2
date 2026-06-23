# tests/rl_mass/test_osim2mjcf_muscles.py
"""osim2mjcf muscles: 6 spatial tendons + muscle actuators with Thelen force."""
import os, sys
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import mujoco
from tools.osim2mjcf import parse_muscles, build_mjcf
from rl_mass.engine import MuJoCoEngine
from rl_mass.finger.tendon_muscle import TendonMuscleSystem
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"
NAMES = {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}

def main():
    root = ET.parse(OSIM).getroot()
    mus = parse_muscles(root)
    assert {m["name"] for m in mus} == NAMES, {m["name"] for m in mus}
    bic = next(m for m in mus if m["name"] == "BIClong")
    assert abs(bic["f0"] - 624.3) < 1e-3, bic["f0"]
    assert len(bic["points"]) == 9, len(bic["points"])  # BIClong has 9 path points

    xml = build_mjcf(OSIM)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nu == 6, m.nu
    anames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)}
    assert anames == NAMES, anames
    # f0 lives in gainprm[:,2] for MuJoCo muscle actuators (what TendonMuscleSystem reads)
    bic_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "BIClong")
    assert abs(m.actuator_gainprm[bic_id, 2] - 624.3) < 1e-3, m.actuator_gainprm[bic_id, 2]
    # MuJoCo auto-computed a valid length range
    assert np.all(m.actuator_lengthrange[:, 1] > m.actuator_lengthrange[:, 0])

    # TendonMuscleSystem builds against the generated model
    d = tempfile.mkdtemp(); p = os.path.join(d, "arm26.xml")
    with open(p, "w") as f: f.write(xml)
    eng = MuJoCoEngine(p, timestep=1/600); eng.forward()
    tms = TendonMuscleSystem(eng, force_scale=1.0)
    assert tms.n_muscles == 6 and tms.num_total_related_dofs > 0
    print("OK")

if __name__ == "__main__":
    main()
