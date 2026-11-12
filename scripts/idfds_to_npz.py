"""Converts the real flights of IDF-DS (Fixed-Wing UAS Telemetry Benchmark, Universitat Politecnica de Valencia,
Zenodo record 16992976, CC BY 4.0; Volantex Ranger 2400 with a Pixhawk 6X running PX4) into the shard format
of scripts/generate_flights.py, so that train_a.py and eval_skyjepa.py run on them unchanged.

Input: the "UnGrouped flights" folders of the Pixhawk archive (one folder per lap, one CSV per PX4 topic).
Topics read (rate in the logs):
  vehicle_attitude_0          quaternion q (body FRD to NED, w first)            20 Hz
  vehicle_angular_velocity_0  body rates xyz                                     50 Hz
  vehicle_local_position_0    x, y, z, vx, vy, vz (NED)                          10 Hz
  airspeed_wind_0             wind estimate windspeed_north / _east              1 Hz
  actuator_servos_0           normalized surfaces: 0 left aileron, 1 right aileron, 2 elevator, 3 rudder   10 Hz
  actuator_motors_0           normalized throttle control[0]                     10 Hz
The servo indices were identified on lap_001 from their correlation with the body rates (aileron pair with p,
elevator with q, rudder with r).

Everything is put on a 20 Hz grid. Continuous signals are interpolated linearly; commands are held (zero-order
hold), as the actuators apply them. The validated airspeed of these logs is empty (the pitot is not fused), so
the air-relative velocity is rebuilt from the estimator: v_air = v_ground - wind (vertical wind 0), rotated into
the body frame; Va = |v_air|, alpha = atan2(w, u), beta = asin(v / Va). All states are onboard estimates (EKF);
there is no external ground truth.

Output: <out>/train_shard000.npz with every lap and its split (train / val / test, 80/10/10 by lap, fixed seed),
so that eval_skyjepa.py --set train evaluates the held-out laps ("vols_de_test"). The aircraft parameters are the
same for every lap (mass from the dataset paper; inertias unknown, set to a rough estimate, used only by the
physics heads).

Usage: python scripts/idfds_to_npz.py --src "<extracted>/Holybro Pixhawk/processed/UnGrouped flights" \
           --out data/data_idfds
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import AIRCRAFT, ORACLE  # noqa: E402

DT = 0.05
TOPICS = ("vehicle_attitude_0", "vehicle_angular_velocity_0", "vehicle_local_position_0", "airspeed_wind_0",
          "actuator_servos_0", "actuator_motors_0")
CHANNELS = ["t", "u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "north", "east", "h", "Va", "alpha", "beta",
            "de_cmd", "da_cmd", "dr_cmd", "dt_cmd", "wind_n_est", "wind_e_est", "wind_d_est", "alive"]
# Ranger 2400 with Pixhawk + Jetson: 2.16 kg (dataset paper). Inertias are not published: rough estimate for a
# 2.4 m span, 2.2 kg high wing (kg m^2), only read by the physics heads.
AIRCRAFT_VALUES = dict(mass=2.16, Ixx=0.25, Iyy=0.15, Izz=0.38, Ixz=0.0)


def read(folder, topic):
    """{column: float array} of a topic CSV, sorted by timestamp without duplicates (empty fields: NaN)."""
    f = sorted(folder.glob(f"*_{topic}.csv"))
    if not f:
        raise FileNotFoundError(f"{folder.name}: no {topic}")
    with open(f[0]) as fh:
        names = fh.readline().strip().split(",")
    a = np.genfromtxt(f[0], delimiter=",", skip_header=1, dtype=np.float64, invalid_raise=False)
    a = np.atleast_2d(a)
    _, keep = np.unique(a[:, names.index("timestamp")], return_index=True)
    a = a[keep]
    return {n: a[:, i] for i, n in enumerate(names)}


def lin(d, col, t):
    return np.interp(t, d["timestamp"] * 1e-6, d[col])


def hold(d, col, t):
    ts = d["timestamp"] * 1e-6
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return d[col][i]


def euler(qw, qx, qy, qz):
    phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    theta = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return phi, theta, psi


def ned_to_body(phi, theta, psi, vn, ve, vd):
    """Body FRD components of an NED vector (R^T v, R = Rz(psi) Ry(theta) Rx(phi))."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    u = ct * cp * vn + ct * sp * ve - st * vd
    v = (sf * st * cp - cf * sp) * vn + (sf * st * sp + cf * cp) * ve + sf * ct * vd
    w = (cf * st * cp + sf * sp) * vn + (cf * st * sp - sf * cp) * ve + cf * ct * vd
    return u, v, w


def lap(folder):
    """(K, len(CHANNELS)) float32 on a 20 Hz grid for one lap folder."""
    d = {k: read(folder, k) for k in TOPICS}
    t0 = max(x["timestamp"][0] for x in d.values()) * 1e-6
    t1 = min(x["timestamp"][-1] for x in d.values()) * 1e-6
    t = np.arange(t0, t1, DT)
    att, av, lp, wd, sv, mo = (d[k] for k in TOPICS)
    q = np.stack([lin(att, f"q[{i}]", t) for i in range(4)])
    q /= np.linalg.norm(q, axis=0, keepdims=True)
    phi, theta, psi = euler(*q)
    p, qr, r = (lin(av, f"xyz[{i}]", t) for i in range(3))
    vn, ve, vd = (lin(lp, c, t) for c in ("vx", "vy", "vz"))
    wn, we = lin(wd, "windspeed_north", t), lin(wd, "windspeed_east", t)
    u, v, w = ned_to_body(phi, theta, psi, vn - wn, ve - we, vd)
    Va = np.sqrt(u * u + v * v + w * w)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(np.clip(v / np.maximum(Va, 1e-3), -1, 1))
    s = [hold(sv, f"control[{i}]", t) for i in range(4)]
    cols = dict(t=t - t0, u=u, v=v, w=w, p=p, q=qr, r=r, phi=phi, theta=theta, psi=psi,
                north=lin(lp, "x", t), east=lin(lp, "y", t), h=-lin(lp, "z", t), Va=Va, alpha=alpha, beta=beta,
                de_cmd=s[2], da_cmd=0.5 * (s[1] - s[0]), dr_cmd=s[3], dt_cmd=hold(mo, "control[0]", t),
                wind_n_est=wn, wind_e_est=we, wind_d_est=np.zeros_like(t))
    X = np.stack([cols[c] if c != "alive" else np.ones_like(t) for c in CHANNELS], 1)
    ok = np.isfinite(X).all(1) & (Va > 8.0)
    ok[:int(1 / DT)] = False                          # first and last second: interpolation edges
    ok[-int(1 / DT):] = False
    X[:, CHANNELS.index("alive")] = ok
    X[~np.isfinite(X)] = 0.0
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the 'UnGrouped flights' folder (one sub-folder per lap)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the 80/10/10 split by lap")
    a = ap.parse_args()
    folders = sorted(f for f in Path(a.src).iterdir() if f.is_dir() and f.name.startswith("lap_"))
    laps, names = [], []
    for f in folders:
        try:
            X = lap(f)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  skipped {f.name}: {e}")
            continue
        laps.append(X)
        names.append(f.name)
        print(f"  {f.name}: {len(X) * DT:6.1f} s, valid {X[:, -1].mean():.2f}, Va {np.median(X[:, CHANNELS.index('Va')]):.1f} m/s",
              flush=True)
    K = max(len(X) for X in laps)
    data = np.zeros((len(laps), K, len(CHANNELS)), np.float32)       # padding: alive = 0
    for i, X in enumerate(laps):
        data[i, :len(X)] = X
    split = np.random.default_rng(a.seed).choice(["train", "val", "test"], size=len(laps), p=[0.8, 0.1, 0.1])
    n = len(laps)
    meta = {k: np.full(n, AIRCRAFT_VALUES[k], np.float32) for k in AIRCRAFT}
    meta.update({k: np.full(n, 0.0 if k == "cg_shift" else 1.0, np.float32) for k in ORACLE})
    meta["is_test_aircraft"] = np.zeros(n, bool)
    meta["lap"] = np.array(names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train_shard000.npz", data=data, channels=np.array(CHANNELS), split=split, **meta)
    hours = data[:, :, -1].sum() * DT / 3600
    print(f"written {out / 'train_shard000.npz'}: {n} laps ({(split == 'train').sum()} train, "
          f"{(split == 'val').sum()} val, {(split == 'test').sum()} test), {hours:.1f} h of valid flight")


if __name__ == "__main__":
    main()
