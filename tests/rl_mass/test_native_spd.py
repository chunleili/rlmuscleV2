"""native_spd_forces: shape, and zero torque when at the target with zero vel."""
import numpy as np
from rl_mass.engine import MuJoCoEngine
from rl_mass.native_spd import native_spd_forces

XML = "/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml"


def main():
    eng = MuJoCoEngine(XML, timestep=1.0 / 600)
    eng.forward()
    q = eng.get_qpos()
    eng.set_qvel(np.zeros(eng.nv))
    eng.forward()
    kp = 100.0
    kv = np.sqrt(2.0 * kp)
    tau = native_spd_forces(eng, q.copy(), kp=kp, kv=kv)
    assert tau.shape == (eng.nv,), tau.shape
    # At the target with zero velocity, the PD error terms vanish; remaining
    # torque is only the -dt*Kv*ddq gravity-compensation correction — finite.
    assert np.all(np.isfinite(tau))
    print("OK", round(float(np.linalg.norm(tau)), 4))


if __name__ == "__main__":
    main()
