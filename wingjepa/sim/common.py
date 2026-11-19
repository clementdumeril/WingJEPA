"""Pieces shared by the aircraft simulators (advanced_plane.py, and the archived Ultra Stick model).

State Y (n, 16): u v w | p q r | phi theta psi | north east h | de da dr (positions) | omega (propeller)
"""
import numpy as np
import yaml

(U, V, W, P, Q, R, PHI, TH, PSI, XN, YE, H, DE, DA, DR, OM) = range(16)
N_STATE = 16


def isa_density(h):
    """1976 standard atmosphere, troposphere."""
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dcm_body_to_ned(phi, th, psi):
    """Matrices (n, 3, 3): v_NED = C @ v_body."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    C = np.empty(phi.shape + (3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = -st, sf * ct, cf * ct
    return C
