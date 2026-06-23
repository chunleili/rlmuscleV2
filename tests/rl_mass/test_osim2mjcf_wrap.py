"""osim2mjcf wrap: wrap geoms emitted; model still compiles with 6 muscles."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco
from tools.osim2mjcf import parse_wraps, build_mjcf
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"

def main():
    root = ET.parse(OSIM).getroot()
    wraps = parse_wraps(root)
    # base owns a WrapCylinder (TRIlongglen); r_humerus owns several.
    assert "base" in wraps and any(w["type"] == "WrapCylinder" for w in wraps["base"])
    assert "r_humerus" in wraps and len(wraps["r_humerus"]) >= 3

    xml = build_mjcf(OSIM)
    assert 'class="wrap"' in xml or "wrapgeom" in xml or "_wrap" in xml, "no wrap geom emitted"
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nq == 2, m.nq
    assert m.nu == 6, m.nu
    # at least one tendon wrap geom present
    assert m.ngeom >= 4, m.ngeom
    print("OK")

if __name__ == "__main__":
    main()
