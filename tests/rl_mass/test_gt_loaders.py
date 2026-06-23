"""OpenSimStoLoader parses the real Arm26 SO activation .sto."""
import numpy as np
from rl_mass.finger.gt import OpenSimStoLoader

STO = ("/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/"
       "StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto")
NAMES = {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}

def main():
    gt = OpenSimStoLoader(STO).load()
    assert set(gt.names) == NAMES, gt.names
    assert gt.time.shape[0] == 121, gt.time.shape
    assert gt.activation.shape == (121, 6), gt.activation.shape
    assert gt.activation.min() >= -1e-6 and gt.activation.max() <= 1.0 + 1e-6
    # column selection + order
    gt2 = OpenSimStoLoader(STO, muscle_columns=["BIClong", "TRIlong"]).load()
    assert gt2.names == ["BIClong", "TRIlong"]
    assert gt2.activation.shape == (121, 2)
    print("OK")

if __name__ == "__main__":
    main()
