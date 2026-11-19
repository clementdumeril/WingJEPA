"""Generate a shard of 6DoF flights and write it as .npz + .json.

Usage: python scripts/generate_flights.py --config configs/data_advanced.yaml --set train --shard 3 --n-shards 20
Each shard has its own seed (seed + 1000 * set index + shard): shards are independent.
Aircraft: the `aircraft` key of the aircraft config named by the data config (aircraft.config) selects the
simulator: advanced_plane (configs/advanced_plane.yaml, wingjepa/sim/advanced_plane.py) or the archived Ultra Stick
120 with the NASA tables (archive/ultrastick/configs/*.yaml, imported only then; prerequisite:
bash archive/ultrastick/scripts/fetch_umn.sh, once).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim import advanced_plane  # noqa: E402
from wingjepa.sim.common import (DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W, XN, YE,  # noqa: E402
                                 dcm_body_to_ned, load_config)
from wingjepa.sim.pilot import LAT_NAMES, TECS, TYPE_NAMES, lateral, make_profiles  # noqa: E402
from wingjepa.sim.wind import Wind  # noqa: E402

CHANNELS = ["t", "u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "north", "east", "h",
            "Va", "alpha", "beta", "gamma_a", "e_spec",
            "de_cmd", "da_cmd", "dr_cmd", "dt_cmd", "de", "da", "dr", "omega_rpm", "J",
            "wind_n", "wind_e", "wind_d", "turb_u", "turb_v", "turb_w", "wind_n_est", "wind_e_est", "wind_d_est",
            "Fx_aero", "Fy_aero", "Fz_aero", "Mx_aero", "My_aero", "Mz_aero", "Mx_prop", "My_prop", "Mz_prop",
            "lift", "drag", "side", "CD", "CY", "CL", "Cl", "Cm", "Cn", "thrust", "P_motor", "P_shaft", "E_motor",
            "phi_ref", "engine_off", "open_loop", "protect", "alive"]
AERO = ("CD", "CY", "CL", "Cl", "Cm", "Cn")


def make_aircraft(rng, n, p, ranges, is_test):
    """Batch of aircraft with per-flight randomization: AdvancedPlane when the aircraft config says
    `aircraft: advanced_plane` (ranges: the data config's aircraft.ranges, else the aircraft config's own
    randomization section), the archived Ultra Stick 120 (NASA tables) when it names an Ultra Stick.
    Returns (simulator, overrides)."""
    if p.get("aircraft") == "advanced_plane":
        ov = advanced_plane.sample_overrides(rng, n, p, ranges, is_test)
        return advanced_plane.AdvancedPlane(p, n, None, ov), ov
    if str(p.get("aircraft", "")).startswith("ultrastick"):
        from archive.ultrastick.ultrastick import FaserTables, UltraStick, sample_overrides
        ov = sample_overrides(rng, n, p, ranges, is_test)
        return UltraStick(p, n, FaserTables(p["aero_file"]), ov), ov
    raise ValueError(f"aircraft config: unknown aircraft {p.get('aircraft')!r} (advanced_plane, ultrastick120)")


def draw_upsets(rng, n, c):
    """Start of the recorded flight away from trim (sim.upset of the data config), for a fraction c["prob"] of
    the flights: bank and pitch added to the trimmed attitude, airspeed scaled (kept in c["V_bounds"]), angle of
    attack and sideslip added, body rates added. Each [lo, hi] is a magnitude with a random sign. Drawn after
    every other draw, so a config without `upset` keeps its random sequence."""
    on = rng.random(n) < c["prob"]
    sym = lambda r: rng.uniform(*r, n) * rng.choice([-1.0, 1.0], n)
    d = dict(dphi=np.deg2rad(sym(c["phi_deg"])), dtheta=np.deg2rad(sym(c["theta_deg"])),
             dalpha=np.deg2rad(sym(c["alpha_deg"])), dbeta=np.deg2rad(sym(c["beta_deg"])),
             dp=np.deg2rad(sym(c["p_dps"])), dq=np.deg2rad(sym(c["q_dps"])), dr=np.deg2rad(sym(c["r_dps"])))
    d = {k: np.where(on, v, 0.0) for k, v in d.items()}
    d["V_factor"] = np.where(on, rng.uniform(*c["V_factor"], n), 1.0)
    d["on"] = on
    return d


def apply_upset(Y, d, V_bounds):
    """Perturbed copy of the state Y (air-relative u, v, w rebuilt from the new Va, alpha, beta)."""
    Y = Y.copy()
    on = d["on"]
    uvw = Y[:, U:W + 1]
    Va = np.linalg.norm(uvw, axis=1)
    al = np.arctan2(uvw[:, 2], uvw[:, 0]) + d["dalpha"]
    be = np.arcsin(np.clip(uvw[:, 1] / Va, -1, 1)) + d["dbeta"]
    Va = np.where(on, np.clip(Va * d["V_factor"], *V_bounds), Va)
    new = np.stack([Va * np.cos(al) * np.cos(be), Va * np.sin(be), Va * np.sin(al) * np.cos(be)], 1)
    Y[:, U:W + 1] = np.where(on[:, None], new, uvw)
    Y[:, PHI] += d["dphi"]
    Y[:, TH] += d["dtheta"]
    Y[:, P] += d["dp"]
    Y[:, Q] += d["dq"]
    Y[:, R] += d["dr"]
    return Y


def simulate(rng, n, duration, n_test, cfg):
    sim, pil = cfg["sim"], cfg["pilot"]
    p = load_config(cfg["aircraft"]["config"])
    is_test = np.r_[np.zeros(n - n_test, bool), np.ones(n_test, bool)]
    ac, ov = make_aircraft(rng, n, p, cfg["aircraft"].get("ranges"), is_test)
    g = p["g"]

    dt_sim, dt_ctrl = sim["dt"], sim["dt_record"]
    sub = int(round(dt_ctrl / dt_sim))
    K_warm, K = int(round(sim["warmup_s"] / dt_ctrl)), int(round(duration / dt_ctrl))
    h0 = rng.uniform(*sim["h0_range"], n)
    V0 = rng.uniform(*pil["V_range"], n)
    psi0 = rng.uniform(-np.pi, np.pi, n)
    Y, cmd0, thr0, trim_ok = ac.trim(V0, h=h0, psi=psi0)
    if not trim_ok.all():   # aircraft cannot be trimmed at V0 (e.g. too little power): retry at 18 m/s
        V0 = np.where(trim_ok, V0, sim.get("trim_retry_V", 18.0))
        Y2, c2, t2, ok2 = ac.trim(V0, h=h0, psi=psi0)
        Y, cmd0, thr0 = np.where(trim_ok[:, None], Y, Y2), np.where(trim_ok[:, None], cmd0, c2), np.where(trim_ok, thr0, t2)
        trim_ok = trim_ok | ok2
    wind = Wind(rng, n, cfg["wind"], sim["warmup_s"] + duration)
    prof = make_profiles(rng, n, K, dt_ctrl, pil, h0, V0)
    tecs = TECS(n, pil["tecs"], thr_ff=thr0, theta_ff=Y[:, TH], de_ff=cmd0[:, 0], g=g)
    lim, prot, lat_g = sim["limits"], pil["protection"], pil["lateral"]["gains"]
    up = sim.get("upset")                          # optional: recovery from an attitude far from trim
    ups = draw_upsets(rng, n, up) if up else None

    alive = trim_ok.copy()
    t_dead = np.where(trim_ok, np.nan, -sim["warmup_s"])
    E_motor, prot_timer = np.zeros(n), np.zeros(n)
    rec = np.zeros((n, K, len(CHANNELS)), np.float32)
    held = np.c_[cmd0, thr0]                       # de, da, dr, throttle
    turb = np.zeros((n, 3))
    wind_est = wind.steady.copy()
    t = 0.0

    def gust_body(Y, t):
        Cbn = np.swapaxes(dcm_body_to_ned(Y[:, PHI], Y[:, TH], Y[:, PSI]), 1, 2)
        gn = wind.gust(t)
        return np.einsum("nij,nj->ni", Cbn, gn) + turb, gn

    for kk in range(K_warm + K):
        k = kk - K_warm
        if k == 0 and ups is not None:             # the upset happens when the recording starts
            Y = np.where(alive[:, None], apply_upset(Y, ups, up["V_bounds"]), Y)
        gb, gn = gust_body(Y, t)
        F = ac.forces(Y, held[:, 3], gb)
        Va, alpha, beta = F["Va"], F["alpha"], F["beta"]
        C = dcm_body_to_ned(Y[:, PHI], Y[:, TH], Y[:, PSI])
        hdot_air = -np.einsum("nj,nj->n", C[:, 2, :], Y[:, U:W + 1])
        gamma = np.arcsin(np.clip(hdot_air / Va, -1, 1))
        phi, th = Y[:, PHI], Y[:, TH]
        psidot = g * np.tan(np.clip(phi, -1.3, 1.3)) / Va                # coordinated turn
        q_ref, r_ref = psidot * np.cos(th) * np.sin(phi), psidot * np.cos(th) * np.cos(phi)

        # Envelope protection: near stall or at an excessive attitude, take over control.
        # True air data (gusts included) smoothed over ~0.5 s, like the filtered sensors of an
        # autopilot: turbulence spikes are ignored, genuine excursions out of the envelope are not.
        if kk == 0:
            air_f = np.stack([Va, alpha, beta], 1)
        if k == 0 and ups is not None:             # the upset developed before the recording: filter settled on it
            air_f = np.where(ups["on"][:, None], np.stack([Va, alpha, beta], 1), air_f)
        air_f += dt_ctrl / (prot["filter_tau_s"] + dt_ctrl) * (np.stack([Va, alpha, beta], 1) - air_f)
        Vm, am, bm = air_f.T
        trig = ((Vm < prot["V_min"]) | (Vm > prot["V_max"])
                | (am > np.deg2rad(prot["alpha_max_deg"])) | (am < np.deg2rad(prot["alpha_min_deg"]))
                | (np.abs(phi) > np.deg2rad(prot["phi_max_deg"])) | (np.abs(th) > np.deg2rad(prot["theta_max_deg"]))
                | (np.abs(bm) > np.deg2rad(prot["beta_max_deg"])) | (Y[:, H] < prot["h_min"]))
        if k >= 0:   # no protection during trim settling (autopilot only)
            prot_timer = np.where(trig, prot["hold_s"], np.maximum(prot_timer - dt_ctrl, 0.0))
        protect = prot_timer > 0

        if k < 0:   # trim settling under autopilot (not recorded)
            h_ref, V_ref, phi_ref = h0, V0, np.zeros(n)
            eoff, freeze = np.zeros(n, bool), np.zeros(n, bool)
            add = np.zeros((n, 4))
        else:
            h_ref, V_ref, phi_ref = prof["h_ref"][:, k], prof["V_ref"][:, k], prof["phi_ref"][:, k]
            eoff, freeze = prof["engine_off"][:, k], prof["open_loop"][:, k]
            add = np.stack([prof["de_add"][:, k], prof["da_add"][:, k], prof["dr_add"][:, k], prof["dt_add"][:, k]], 1)
            h_ref = np.where(protect, np.maximum(Y[:, H], prot["h_min"] + 40.0), h_ref)
            V_ref = np.where(protect, np.maximum(V_ref, prot["V_target"]), V_ref)
            phi_ref = np.where(protect, 0.0, phi_ref)
            eoff, freeze = eoff & ~protect, freeze & ~protect
            add = np.where(protect[:, None], 0.0, add)

        # bank angle limited at low speed (the load factor of a turn brings the aircraft closer to stall):
        # bank_min_lim_deg up to bank_lim_V[0], rising to bank_deg[1] at bank_lim_V[1] (default 15 -> 23 m/s)
        V_lo, V_hi = pil["lateral"].get("bank_lim_V", (15.0, 23.0))
        bank_max = np.deg2rad(pil["lateral"]["bank_min_lim_deg"]) + np.deg2rad(
            pil["lateral"]["bank_deg"][1] - pil["lateral"]["bank_min_lim_deg"]) * np.clip((Va - V_lo) / (V_hi - V_lo), 0, 1)
        phi_ref = np.clip(phi_ref, -bank_max, bank_max)
        de_c, thr_c = tecs(Va, am, gamma, th, Y[:, Q], q_ref, Y[:, H], h_ref, V_ref, eoff, freeze, dt_ctrl)
        da_c, dr_c = lateral(phi_ref, phi, Y[:, P], Y[:, R], r_ref, beta, lat_g, cmd0[:, 1], cmd0[:, 2])
        held = np.where(freeze[:, None], held, np.stack([de_c, da_c, dr_c, thr_c], 1))   # open loop: frozen
        cmd = np.clip(held[:, :3] + add[:, :3], p["act_min"], p["act_max"])
        thr = np.where(eoff, 0.0, np.clip(held[:, 3] + add[:, 3], 0.0, 1.0))

        F = ac.forces(Y, thr, gb)
        if k >= 0:
            wn = wind.steady + gn
            cols = dict(t=k * dt_ctrl, u=Y[:, U], v=Y[:, V], w=Y[:, W], p=Y[:, P], q=Y[:, Q], r=Y[:, R],
                        phi=phi, theta=th, psi=Y[:, PSI], north=Y[:, XN], east=Y[:, YE], h=Y[:, H],
                        Va=Va, alpha=alpha, beta=beta, gamma_a=gamma, e_spec=Y[:, H] + Va**2 / (2 * g),
                        de_cmd=cmd[:, 0], da_cmd=cmd[:, 1], dr_cmd=cmd[:, 2], dt_cmd=thr,
                        de=Y[:, DE], da=Y[:, DA], dr=Y[:, DR], omega_rpm=Y[:, OM] * 30 / np.pi, J=F["J"],
                        wind_n=wn[:, 0], wind_e=wn[:, 1], wind_d=wn[:, 2],
                        turb_u=turb[:, 0], turb_v=turb[:, 1], turb_w=turb[:, 2],
                        wind_n_est=wind_est[:, 0], wind_e_est=wind_est[:, 1], wind_d_est=wind_est[:, 2],
                        Fx_aero=F["F_aero"][:, 0], Fy_aero=F["F_aero"][:, 1], Fz_aero=F["F_aero"][:, 2],
                        Mx_aero=F["M_aero"][:, 0], My_aero=F["M_aero"][:, 1], Mz_aero=F["M_aero"][:, 2],
                        Mx_prop=F["M_prop"][:, 0], My_prop=F["M_prop"][:, 1], Mz_prop=F["M_prop"][:, 2],
                        lift=F["lift"], drag=F["drag"], side=F["side"],
                        **{c: F["C"][:, i] for i, c in enumerate(AERO)},
                        thrust=F["T"], P_motor=F["P_motor"], P_shaft=F["P_shaft"], E_motor=E_motor,
                        phi_ref=phi_ref, engine_off=eoff, open_loop=freeze, protect=protect, alive=alive)
            rec[:, k, :] = np.stack([np.broadcast_to(cols[c], (n,)) for c in CHANNELS], 1)

        for _ in range(sub):
            turb = wind.step_turbulence(ac.forces(Y, thr, gust_body(Y, t)[0])["Va"], Y[:, H], dt_sim)
            gb, gn = gust_body(Y, t)
            wind_est = wind.step_estimate(wind.steady + gn, dt_sim)
            Yn = ac.step(Y, cmd, thr, gb, wind.steady, dt_sim)
            # envelope checked at every integration step: a flight that leaves it is frozen at its last valid state
            with np.errstate(all="ignore"):
                vr = Yn[:, U:W + 1] - gb
                Va_n = np.linalg.norm(vr, axis=1)
                a_n = np.arctan2(vr[:, 2], vr[:, 0])
                b_n = np.arcsin(np.clip(vr[:, 1] / np.maximum(Va_n, 1e-3), -1, 1))
                bad = (~np.isfinite(Yn).all(1) | (Va_n < lim["Va_min"]) | (Va_n > lim["Va_max"])
                       | (a_n > np.deg2rad(lim["alpha_max_deg"])) | (a_n < np.deg2rad(lim["alpha_min_deg"]))
                       | (np.abs(b_n) > np.deg2rad(lim["beta_max_deg"]))
                       | (np.abs(Yn[:, PHI]) > np.deg2rad(lim["phi_max_deg"]))
                       | (np.abs(Yn[:, TH]) > np.deg2rad(lim["theta_max_deg"])) | (Yn[:, H] < lim["h_min"]))
            t_dead[bad & alive] = t - sim["warmup_s"]
            alive &= ~bad
            Y = np.where(alive[:, None], Yn, Y)
            t += dt_sim
        E_motor += np.where(alive, F["P_motor"] * dt_ctrl, 0.0)

    meta = dict(type=prof["types"], lat_mode=prof["lat"], is_test_aircraft=is_test, h0=h0, V0=V0,
                trim_ok=trim_ok, mass=ov["mass"], Ixx=ov["Ixx"], Iyy=ov["Iyy"], Izz=ov["Izz"], Ixz=ov["Ixz"],
                power_scale=ov["power_scale"], cg_shift=ov["cg_shift"],
                **{f"aero_scale_{k}": v for k, v in ov["aero_scale"].items()},     # AERO (+ Cmq, de, da, dr)
                **{k: ov[k] for k in ("servo_tau",) if k in ov},                    # advanced_plane only
                wind_steady_n=wind.steady[:, 0], wind_steady_e=wind.steady[:, 1], wind_steady_d=wind.steady[:, 2],
                turb_level=wind.turb_level, t_dead=t_dead)
    if ups is not None:
        meta.update(upset=ups["on"], **{f"upset_{k}": v for k, v in ups.items() if k != "on"})
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
                                  f"{int((meta['type'] == i).sum())}" for i in range(4)},
                   alive_by_lateral={LAT_NAMES[i]: f"{int(alive_end[meta['lat_mode'] == i].sum())}/"
                                     f"{int((meta['lat_mode'] == i).sum())}" for i in range(3)})
    json.dump(summary, open(stem.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
