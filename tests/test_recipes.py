"""Every recipe of the SkyJEPA test suite on the advanced_plane, built from the command line of scripts/train_a.py
(the flags of the cluster runs) with the advanced_plane prior, then one forward pass on random data. No training."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_a import REMOVED, build_parser, resolve  # noqa: E402
from wingjepa.data.windows import FEATURES, N_STATE, Normalizer  # noqa: E402
from wingjepa.eval.io import load_model  # noqa: E402
from wingjepa.models.jepa import model_from_config  # noqa: E402

PRIOR = str(ROOT / "configs/priors/advanced_plane_nul.json")
ARCH = ("--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1 --T 20 "
        "--batch 2048 --warmup 4000 --steps1 24000 --sigreg-steps 20 --device cuda").split()
RECIPES = {   # run name -> (options after --mode, trunk run or None)
    "a0": ("a0", None),
    "a0_pred": ("a0 --T 1 --T-max 1", None),
    "a0_physreg": ("a0 --T 1 --T-max 1 --lambda-phys 1", None),
    "a0_noiseaug": ("a0 --input-noise 0.5", None),
    "jepa": ("jepa --lambda-sig 0.02", None),
    "c_jepa_a0": (f"c_jepa --prior {PRIOR}", "a0"),
    "c_jepa_jepa": (f"c_jepa --prior {PRIOR}", "jepa"),
    "c_direct": (f"c_direct --prior {PRIOR}", None),
    "mlp_direct": ("mlp_direct --mlp-hidden 50,50", None),
}
# normalization around the advanced_plane cruise (15 m/s), so that the physics head integrates plausible states
MEAN = np.array([15, 0.05, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0.25, 0, 0, 0], np.float32)
STD = np.array([2, 0.05, 0.03, 0.3, 0.3, 0.3, 0.2, 0.1, 1, 0.05, 0.05, 0.05, 0.1, 1, 1, 1], np.float32)
AIRCRAFT = [1.005, 0.197563975, 0.146509604, 0.148317604, 0.0]   # configs/advanced_plane.yaml mass, inertias


def build(name, runs):
    """Parse the recipe's command line, rebuild its model as train_a.py does and save it as a run (config.json,
    model.pt) under `runs`, so that a c_jepa recipe can take it as its trunk."""
    opts, trunk = RECIPES[name]
    argv = ["--data-dir", "unused", "--set", "train", "--out", str(runs / name), "--seed", "0", *ARCH,
            "--mode", *opts.split()]
    if trunk:
        argv += ["--trunk", str(runs / trunk)]
    ap = build_parser()
    a = ap.parse_args(argv)
    trunk_cfg = resolve(ap, a)
    torch.manual_seed(a.seed)
    norm = Normalizer(MEAN, STD)
    model = model_from_config(dict(vars(a), norm=norm.to_dict()))
    if trunk_cfg:                                                      # frozen trunk, as in train_a.main
        tm = model_from_config(trunk_cfg)
        tm.load_state_dict(torch.load(runs / trunk / "model.pt"))
        model.encoder.load_state_dict(tm.encoder.state_dict())
        model.predictor.load_state_dict(tm.predictor.state_dict())
    (runs / name).mkdir(parents=True)
    json.dump(dict(vars(a), features=FEATURES, n_params=model.n_params(), norm=norm.to_dict()),
              open(runs / name / "config.json", "w"))
    torch.save(model.state_dict(), runs / name / "model.pt")
    return a, model


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    return d, {name: build(name, d) for name in RECIPES}               # trunks (a0, jepa) come before c_jepa


@pytest.mark.parametrize("name", list(RECIPES))
def test_recipe_builds_and_predicts(runs, name):
    d, built = runs
    a, model = built[name]
    assert a.head == ("physics" if a.mode in ("c_direct", "c_jepa") else "mlp")
    for k, off in REMOVED.items():                                     # config.json keys of the removed options
        assert getattr(a, k) == off
    torch.manual_seed(1)
    B, T = 4, a.T_max or a.T
    seq = 0.5 * torch.randn(B, a.H + T, 16)
    ac = torch.tensor([AIRCRAFT] * B)
    with torch.no_grad():
        pred, raw = model.predict_states(seq, a.H, True, ac)
    assert pred.shape == (B, T, N_STATE) and torch.isfinite(pred).all()
    if a.head == "physics":
        assert raw.shape == (B, T, 6)
    if a.mode == "jepa":
        assert model.target_latents(seq, a.H).shape == (B, T, 24)
    if a.mode == "c_jepa":                                             # the trunk is the one of its run
        trunk = built[RECIPES[name][1]][1]
        torch.testing.assert_close(model.encoder.state_dict(), trunk.encoder.state_dict())
        assert a.H == 11 and a.action_encoder == 1
    m2, cfg, _ = load_model(d / name)                                  # reloads as the evaluation scripts do
    with torch.no_grad():
        torch.testing.assert_close(m2.predict_states(seq, cfg["H"], True, ac)[0], pred)
