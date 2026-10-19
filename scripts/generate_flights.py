"""Génère un paquet (shard) de vols de l'Ultra Stick 120 et l'écrit en .npz + .json.

Usage : python scripts/generate_flights.py --config configs/data_v1.yaml --set train --shard 3 --n-shards 20
Chaque shard a sa propre graine (seed + 1000 * index du set + shard) : les shards sont indépendants.
Prérequis : bash scripts/fetch_umn.sh (une fois).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.pilot import TECS, TYPE_NAMES, make_profiles  # noqa: E402
from wingjepa.sim.ultrastick import DE, H, OM, Q, TH, U, W, X, FaserTables, UltraStick, load_config  # noqa: E402
from wingjepa.sim.wind import Wind  # noqa: E402

CHANNELS = ["t", "u", "w", "q", "theta", "x", "h", "Va", "alpha", "gamma_a", "e_spec",
            "delta_e_cmd", "delta_t_cmd", "delta_e", "omega_rpm", "J",
            "wind_x", "wind_d", "turb_u", "turb_w", "wind_x_est", "wind_d_est",
            "Fx_aero", "Fz_aero", "My_aero", "lift", "drag", "CL", "CD", "Cm", "thrust",
            "P_motor", "P_shaft", "E_motor", "engine_off", "open_loop", "alive"]


def _factor(rng, n, r, test):
    if not test:
        return 1 + rng.uniform(-r[0], r[0], n)
    return 1 + rng.choice([-1, 1], n) * rng.uniform(r[0], r[1], n)


def sample_overrides(rng, n, p, ranges, is_test):
    """Randomisation de domaine par vol (facteurs multiplicatifs, ou décalage pour le centrage)."""
    out = {"mass": [], "Iyy": [], "power_scale": [], "cg_shift": [], "CD": [], "CL": [], "Cm": []}
    for test in (False, True):
        m = int((is_test == test).sum())
        out["mass"].append(p["mass"] * _factor(rng, m, ranges["mass"], test))
        out["Iyy"].append(p["Iyy"] * _factor(rng, m, ranges["Iyy"], test))
        out["power_scale"].append(_factor(rng, m, ranges["motor"], test))
        out["cg_shift"].append(_factor(rng, m, ranges["cg_shift"], test) - 1)
        for k in ("CD", "CL", "Cm"):
            out[k].append(_factor(rng, m, ranges["aero"], test))
    o = {k: np.concatenate(v) for k, v in out.items()}
    return dict(mass=o["mass"], Iyy=o["Iyy"], power_scale=o["power_scale"], cg_shift=o["cg_shift"],
                aero_scale={k: o[k] for k in ("CD", "CL", "Cm")})


def simulate(rng, n, duration, n_test, cfg):
    sim, pil = cfg["sim"], cfg["pilot"]
    p = load_config(cfg["aircraft"]["config"])
    tab = FaserTables(p["aero_file"])
    is_test = np.r_[np.zeros(n - n_test, bool), np.ones(n_test, bool)]
    ov = sample_overrides(rng, n, p, cfg["aircraft"]["ranges"], is_test)
    ac = UltraStick(p, n, tab, ov)

    dt_sim, dt_ctrl = sim["dt"], sim["dt_record"]
    sub = int(round(dt_ctrl / dt_sim))
    K_warm, K = int(round(sim["warmup_s"] / dt_ctrl)), int(round(duration / dt_ctrl))
    h0 = rng.uniform(*sim["h0_range"], n)
    V0 = rng.uniform(*pil["V_range"], n)
    Y, de0, thr0, trim_ok = ac.trim(V0, h=h0)
    if not trim_ok.all():   # avion impossible à équilibrer à V0 (ex. trop peu de puissance) : on réessaie à 18 m/s
        V0 = np.where(trim_ok, V0, 18.0)
        Y2, de2, thr2, ok2 = ac.trim(V0, h=h0)
        Y, de0, thr0 = np.where(trim_ok[:, None], Y, Y2), np.where(trim_ok, de0, de2), np.where(trim_ok, thr0, thr2)
        trim_ok = trim_ok | ok2
    wind = Wind(rng, n, cfg["wind"], sim["warmup_s"] + duration)

    prof = make_profiles(rng, n, K, dt_ctrl, pil, h0, V0)
    tecs = TECS(n, pil["tecs"], thr_ff=thr0, theta_ff=Y[:, TH], de_ff=de0)
    lim = sim["limits"]
    alive = trim_ok.copy()
    t_dead = np.where(trim_ok, np.nan, -sim["warmup_s"])
    E_motor = np.zeros(n)
    rec = np.zeros((n, K, len(CHANNELS)), np.float32)
    held_e, held_t = de0.copy(), thr0.copy()
    ug_t, wg_t = np.zeros(n), np.zeros(n)
    wx_est, wd_est = wind.steady_x.copy(), wind.steady_d.copy()
    t = 0.0

    def unsteady(Y, t):
        """Rafales (inertiel -> axes avion) + turbulence (déjà en axes avion)."""
        wx, wd = wind.inertial(t)
        gx, gd = wx - wind.steady_x, wd - wind.steady_d
        th = Y[:, TH]
        return gx * np.cos(th) - gd * np.sin(th) + ug_t, gx * np.sin(th) + gd * np.cos(th) + wg_t, wx, wd

    for kk in range(K_warm + K):
        k = kk - K_warm
        ug, wg, wx, wd = unsteady(Y, t)
        F = ac.forces(Y, held_t, ug, wg)
        gamma_a = Y[:, TH] - F["alpha"]
        if k < 0:   # mise en équilibre sous autopilote (non enregistrée)
            h_ref, V_ref, eoff = h0, V0, np.zeros(n, bool)
            freeze, de_add, dt_add = np.zeros(n, bool), 0.0, 0.0
        else:
            h_ref, V_ref, eoff = prof["h_ref"][:, k], prof["V_ref"][:, k], prof["engine_off"][:, k]
            freeze, de_add, dt_add = prof["open_loop"][:, k], prof["de_add"][:, k], prof["dt_add"][:, k]
        de_c, thr_c = tecs(F["Va"], gamma_a, Y[:, TH], Y[:, Q], Y[:, H], h_ref, V_ref, eoff, freeze, dt_ctrl)
        held_e = np.where(freeze, held_e, de_c)            # boucle ouverte : commandes figées
        held_t = np.where(freeze, held_t, thr_c)
        cmd_e = np.clip(held_e + de_add, p["elev_min"], p["elev_max"])
        cmd_t = np.where(eoff, 0.0, np.clip(held_t + dt_add, 0.0, 1.0))

        F = ac.forces(Y, cmd_t, ug, wg)
        if k >= 0:
            cols = dict(t=np.full(n, k * dt_ctrl), u=Y[:, U], w=Y[:, W], q=Y[:, Q], theta=Y[:, TH],
                        x=Y[:, X], h=Y[:, H], Va=F["Va"], alpha=F["alpha"], gamma_a=Y[:, TH] - F["alpha"],
                        e_spec=Y[:, H] + F["Va"] ** 2 / (2 * p["g"]), delta_e_cmd=cmd_e, delta_t_cmd=cmd_t,
                        delta_e=Y[:, DE], omega_rpm=Y[:, OM] * 30 / np.pi, J=F["J"],
                        wind_x=wx, wind_d=wd, turb_u=ug_t, turb_w=wg_t, wind_x_est=wx_est, wind_d_est=wd_est,
                        Fx_aero=F["Fx"], Fz_aero=F["Fz"], My_aero=F["My"], lift=F["lift"], drag=F["drag"],
                        CL=F["CL"], CD=F["CD"], Cm=F["Cm"], thrust=F["T"], P_motor=F["P_motor"],
                        P_shaft=F["P_shaft"], E_motor=E_motor, engine_off=eoff, open_loop=freeze, alive=alive)
            rec[:, k, :] = np.stack([np.broadcast_to(cols[c], (n,)) for c in CHANNELS], 1)

        for _ in range(sub):
            Va = ac.forces(Y, cmd_t, *unsteady(Y, t)[:2])["Va"]
            ug_t, wg_t = wind.step_turbulence(Va, Y[:, H], dt_sim)
            ug, wg, wx, wd = unsteady(Y, t)
            wx_est, wd_est = wind.step_estimate(wx, wd, dt_sim)
            Yn = ac.step(Y, cmd_e, cmd_t, ug, wg, wind.steady_x, wind.steady_d, dt_sim)
            # enveloppe vérifiée à chaque pas d'intégration : un vol qui en sort est figé sur son
            # dernier état valide (sinon il peut diverger avant le contrôle suivant)
            with np.errstate(all="ignore"):
                Va_n = np.hypot(Yn[:, U] - ug, Yn[:, W] - wg)
                a_n = np.arctan2(Yn[:, W] - wg, Yn[:, U] - ug)
                bad = (~np.isfinite(Yn).all(1) | (Va_n < lim["Va_min"]) | (Va_n > lim["Va_max"])
                       | (a_n > np.deg2rad(lim["alpha_max_deg"])) | (a_n < np.deg2rad(lim["alpha_min_deg"]))
                       | (np.abs(Yn[:, TH]) > np.deg2rad(lim["theta_max_deg"])) | (Yn[:, H] < lim["h_min"]))
            t_dead[bad & alive] = t - sim["warmup_s"]
            alive &= ~bad
            Y = np.where(alive[:, None], Yn, Y)
            t += dt_sim
        E_motor += np.where(alive, F["P_motor"] * dt_ctrl, 0.0)

    meta = dict(type=prof["types"], is_test_aircraft=is_test, h0=h0, V0=V0, trim_ok=trim_ok,
                mass=ov["mass"], Iyy=ov["Iyy"], power_scale=ov["power_scale"], cg_shift=ov["cg_shift"],
                aero_scale_CD=ov["aero_scale"]["CD"], aero_scale_CL=ov["aero_scale"]["CL"],
                aero_scale_Cm=ov["aero_scale"]["Cm"], wind_steady_x=wind.steady_x,
                wind_steady_d=wind.steady_d, turb_level=wind.turb_level, t_dead=t_dead)
    return rec, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    s = cfg["sets"][a.set]
    seed = cfg["seed"] + 1000 * list(cfg["sets"]).index(a.set) + a.shard
    rng = np.random.default_rng(seed)
    n = len(range(a.shard, s["n_flights"], a.n_shards))
    n_test = int(round(n * s.get("test_aircraft_fraction", 0.0)))

    t0 = time.time()
    rec, meta = simulate(rng, n, s["duration_s"], n_test, cfg)
    elapsed = time.time() - t0

    split = rng.choice(["train", "val", "test"], size=n, p=cfg["split"]) if a.set == "train" \
        else np.full(n, "test")
    out = Path(a.out or cfg["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.set}_shard{a.shard:03d}"
    np.savez_compressed(stem.with_suffix(".npz"), data=rec, channels=np.array(CHANNELS), split=split, **meta)

    alive_end = rec[:, -1, CHANNELS.index("alive")] > 0.5
    summary = dict(config=a.config, set=a.set, shard=a.shard, seed=seed, n_flights=n,
                   n_test_aircraft=n_test, duration_s=s["duration_s"], seconds=round(elapsed, 1),
                   trim_ok=int(meta["trim_ok"].sum()), alive_at_end=int(alive_end.sum()),
                   alive_by_type={TYPE_NAMES[i]: f"{int(alive_end[meta['type'] == i].sum())}/"
                                  f"{int((meta['type'] == i).sum())}" for i in range(4)})
    json.dump(summary, open(stem.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
