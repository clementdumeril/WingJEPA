"""Phase 0 : vol plané stabilisé des planeurs JSBSim, pour plusieurs positions de profondeur.

Pour chaque calage de profondeur, on vole 120 s et on moyenne les 40 dernières secondes :
vitesse, taux de chute, finesse, et amplitude des oscillations (phugoïde non amortie ?).
"""
import numpy as np
import jsbsim

FT = 0.3048
DT = 1 / 120


def glide(aircraft, v0_kts, elev, seconds=120.0, alt_ft=6000.0):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model(aircraft)
    fdm.set_dt(DT)
    fdm["ic/h-sl-ft"] = alt_ft
    fdm["ic/vc-kts"] = v0_kts
    fdm["ic/gamma-deg"] = -2.0
    fdm.run_ic()
    fdm["fcs/elevator-cmd-norm"] = elev
    n = int(seconds / DT)
    keep = int(40 / DT)
    V, hd, al = [], [], []
    for i in range(n):
        fdm.run()
        if i >= n - keep:
            V.append(fdm["velocities/vt-fps"] * FT)
            hd.append(fdm["velocities/h-dot-fps"] * FT)
            al.append(fdm["aero/alpha-deg"])
    V, hd, al = map(np.array, (V, hd, al))
    alt = fdm["position/h-agl-ft"] * FT
    sink = -hd.mean()
    return dict(elev=elev, V=V.mean(), sink=sink, LD=V.mean() / sink if sink > 0.05 else float("nan"),
                V_osc=V.std(), alpha=al.mean(), alt_end=alt)


if __name__ == "__main__":
    for ac, v in (("minisgs", 25), ("SGS", 45), ("sgs233", 50), ("sgs126", 50)):
        print(f"--- {ac}")
        for e in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.15):
            try:
                r = glide(ac, v, e)
                print("  elev {elev:+.2f}  V {V:5.1f} m/s  sink {sink:5.2f} m/s  finesse {LD:5.1f}  "
                      "oscill. V {V_osc:4.1f}  alpha {alpha:5.1f} deg  alt fin {alt_end:6.0f} m".format(**r))
            except Exception as ex:  # noqa: BLE001
                print("  erreur", ex)
