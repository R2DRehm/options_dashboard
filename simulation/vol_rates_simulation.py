# simulation/vol_rates_simulation.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .correlated_paths import PathParams

def generate_paths_heston2f(S0, V0, r0, params: PathParams, T, Nf, Nc, seed=None):
    """Heston 2 facteurs (somme de 2 CIR) pour clusters de vol réalistes."""
    rng = np.random.default_rng(seed)
    Nf = int(Nf); Nc = int(Nc); T = float(T)
    dt = T / Nf; sqrt_dt = np.sqrt(dt)

    # facteur "lent" = params de l'UI, facteur "rapide" = plus nerveux
    k2, th2, xi2 = float(params.kappa_v), float(params.theta_v), float(params.xi)
    k1 = max(3.0 * k2, 0.6); th1 = th2; xi1 = max(1.5 * xi2, 0.3)
    w1, w2 = 0.35, 0.65
    rho = float(params.rho)

    kr, thr, sr = float(params.kappa_r), float(params.theta_r), float(params.sigma_r)
    mu = float(params.mu)

    t = np.linspace(0.0, T, Nf + 1)
    S = np.empty(Nf + 1); V1 = np.empty(Nf + 1); V2 = np.empty(Nf + 1); V = np.empty(Nf + 1); r = np.empty(Nf + 1)
    S[0], r[0] = float(S0), float(r0)
    v0 = float(V0)
    V1[0], V2[0] = max(v0 * w1, 1e-10), max(v0 * w2, 1e-10)
    V[0] = w1 * V1[0] + w2 * V2[0]

    Z1 = rng.standard_normal(Nf)
    Z2 = rng.standard_normal(Nf)
    Z3 = rng.standard_normal(Nf)
    Zr = rng.standard_normal(Nf)

    # dW_S corrélé aux deux facteurs
    c1 = rho * np.sqrt(w1)
    c2 = rho * np.sqrt(w2)
    c3 = np.sqrt(max(1.0 - rho * rho, 0.0))

    for i in range(Nf):
        Vi1 = max(V1[i], 0.0)
        V1[i+1] = max(Vi1 + k1 * (th1 - Vi1) * dt + xi1 * np.sqrt(Vi1) * sqrt_dt * Z1[i], 0.0)

        Vi2 = max(V2[i], 0.0)
        V2[i+1] = max(Vi2 + k2 * (th2 - Vi2) * dt + xi2 * np.sqrt(Vi2) * sqrt_dt * Z2[i], 0.0)

        V[i+1] = max(w1 * V1[i+1] + w2 * V2[i+1], 1e-12)

        ri = max(r[i], 0.0)
        r[i+1] = max(ri + kr * (thr - ri) * dt + sr * np.sqrt(ri) * sqrt_dt * Zr[i], 0.0)

        Zs = c1 * Z1[i] + c2 * Z2[i] + c3 * Z3[i]
        S[i+1] = S[i] * np.exp((mu - 0.5 * V[i+1]) * dt + np.sqrt(V[i+1]) * sqrt_dt * Zs)

    coarse_idx = np.linspace(0, Nf, Nc + 1, dtype=int)
    return {"t_fine": t, "S": S, "V": V, "r": r, "coarse_idx": coarse_idx, "rho": rho}

def generate_paths_garch(S0, V0, r0, params: PathParams, T, Nf, Nc, seed=None):
    """GARCH(1,1) sur les rendements, r(t) en Vasicek indépendant."""
    rng = np.random.default_rng(seed)
    Nf = int(Nf); Nc = int(Nc); T = float(T)
    dt = T / Nf; sqrt_dt = np.sqrt(dt)

    # Choix "classiques" GARCH (clustering fort)
    alpha, beta = 0.06, 0.92
    long_var = float(params.theta_v)
    omega = max(long_var * (1.0 - alpha - beta), 1e-10)

    mu = float(params.mu)
    kr, thr, sr = float(params.kappa_r), float(params.theta_r), float(params.sigma_r)

    t = np.linspace(0.0, T, Nf + 1)
    S = np.empty(Nf + 1); V = np.empty(Nf + 1); r = np.empty(Nf + 1)
    S[0], r[0], V[0] = float(S0), float(r0), float(V0)

    Zs = rng.standard_normal(Nf)
    Zr = rng.standard_normal(Nf)

    eps_prev = 0.0
    for i in range(Nf):
        sigma2 = max(omega + alpha * (eps_prev**2) + beta * V[i], 1e-12)
        V[i+1] = sigma2

        ret = (mu - 0.5 * sigma2) * dt + np.sqrt(sigma2) * sqrt_dt * Zs[i]
        S[i+1] = S[i] * np.exp(ret)
        eps_prev = np.sqrt(sigma2) * sqrt_dt * Zs[i]

        ri = max(r[i], 0.0)
        r[i+1] = max(ri + kr * (thr - ri) * dt + sr * np.sqrt(ri) * sqrt_dt * Zr[i], 0.0)

    coarse_idx = np.linspace(0, Nf, Nc + 1, dtype=int)
    return {"t_fine": t, "S": S, "V": V, "r": r, "coarse_idx": coarse_idx, "rho": float(params.rho)}
