"""Phase 0 : vérifications de JSBSim (vitesse, vol plané des planeurs fournis, moteur coupé).

Usage : python scripts/phase0_jsbsim_checks.py
"""
import time

import jsbsim

FT = 0.3048
DT = 1 / 120  # pas de JSBSim par défaut


def make(aircraft, v0_kts, alt_ft=3000.0, gamma_deg=-3.0):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model(aircraft)
    fdm.set_dt(DT)
    fdm["ic/h-sl-ft"] = alt_ft
    fdm["ic/vc-kts"] = v0_kts
    fdm["ic/gamma-deg"] = gamma_deg
    fdm["ic/psi-true-deg"] = 0.0
    fdm.run_ic()
    return fdm


def glide_test(aircraft, v0_kts, seconds=60.0):
    """Vol plané gouvernes au neutre : vitesse, taux de chute et finesse en fin de vol."""
    fdm = make(aircraft, v0_kts)
    fdm["fcs/elevator-cmd-norm"] = 0.0
    n = int(seconds / DT)
    for _ in range(n):
        fdm.run()
    vt = fdm["velocities/vt-fps"] * FT
    hdot = fdm["velocities/h-dot-fps"] * FT
    return {
        "V (m/s)": round(vt, 1),
        "sink (m/s)": round(-hdot, 2),
        "finesse": round(vt / max(-hdot, 1e-3), 1),
        "alpha (deg)": round(fdm["aero/alpha-deg"], 1),
        "theta (deg)": round(fdm["attitude/theta-deg"], 1),
        "masse (kg)": round(fdm["inertia/mass-slugs"] * 14.5939, 1),
    }


def throughput(aircraft, v0_kts, steps=20000):
    fdm = make(aircraft, v0_kts)
    t0 = time.perf_counter()
    for _ in range(steps):
        fdm.run()
    el = time.perf_counter() - t0
    return steps / el


if __name__ == "__main__":
    for ac, v in (("minisgs", 30), ("SGS", 45), ("sgs233", 50), ("sgs126", 50)):
        try:
            print(f"{ac:8s}", glide_test(ac, v))
        except Exception as e:  # noqa: BLE001
            print(f"{ac:8s} ERREUR : {e}")
    sps = throughput("minisgs", 30)
    print(f"vitesse JSBSim (minisgs, 1 processus) : {sps:,.0f} pas/s = {sps * DT:,.0f} s de vol par seconde")
    # Moteur coupé sur un avion motorisé
    fdm = make("c172p", 90)
    fdm["propulsion/set-running"] = 0
    fdm["fcs/throttle-cmd-norm"] = 0.0
    for _ in range(int(30 / DT)):
        fdm.run()
    print("c172p moteur coupé :", {"poussée (lbf)": round(fdm["propulsion/engine/thrust-lbs"], 2),
                                    "sink (m/s)": round(-fdm["velocities/h-dot-fps"] * FT, 2),
                                    "V (m/s)": round(fdm["velocities/vt-fps"] * FT, 1)})
