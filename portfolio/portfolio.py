"""portfolio/portfolio.py – Gestion dynamique de portefeuille d’options

Principales classes :
    • OptionOrder   – décrit un ordre passé sur le grid discret.
    • Portfolio     – agrège les ordres et calcule PnL & grecques au snapshot voulu.

Dépendances :
    - simulation.correlated_paths.generate_paths
    - options.option_pricing (grecs, pricing)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from options.option_pricing import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
)


def _maturity_remaining(T0: float, t_current: float) -> float:
    """Maturité résiduelle non‑négative."""
    return max(T0 - t_current, 0.0)


@dataclass
class OptionOrder:
    """Représente un ordre passé à un snapshot discret."""

    time_idx: int  # indice dans coarse grid
    option_type: str  # "call" | "put"
    strike: float
    quantity: float  # nb contrats (+ long, − short)
    maturity: float  # constante en années depuis t=0


class Portfolio:
    """Portefeuille d’OptionOrder sur une trajectoire simulée."""

    def __init__(self, path_dict: dict[str, np.ndarray]):
        self.path = path_dict  # dict retourné par generate_paths
        self.orders: List[OptionOrder] = []

    # ------------------------------------------------------------------
    # Gestion des ordres
    # ------------------------------------------------------------------
    def add_order(self, order: OptionOrder):
        self.orders.append(order)

    # ------------------------------------------------------------------
    # Valeur + grecques au snapshot courant
    # ------------------------------------------------------------------
    def state_at(self, idx: int) -> pd.DataFrame:
        """Calcule la valeur et les grecques du portefeuille au snapshot `idx`."""
        if not self.orders:
            return pd.DataFrame()

        t_grid = self.path["t_fine"]
        coarse_idx = self.path["coarse_idx"]
        # index fin correspondant au coarse snapshot
        j = coarse_idx[idx]
        t_now = t_grid[j]
        S = self.path["S"][j]
        r = self.path["r"][j]
        sigma = np.sqrt(self.path["V"][j])

        records = []
        for order in self.orders:
            if order.time_idx > idx:
                # ordre pas encore actif
                continue

                        # ----- Sous-jacent simple -----
            if order.option_type == "underlying":
                price = S * order.quantity
                records.append({
                    "option_type": "Underlying",
                    "strike": "-",
                    "qty": order.quantity,
                    "price": price,
                    "delta": order.quantity,  # Δ = 1
                    "gamma": 0.0,
                    "vega":  0.0,
                    "theta": 0.0,
                    "rho":   0.0,
                })
                continue  # passe au prochain ordre


            tau = _maturity_remaining(order.maturity, t_now)
            price = bs_price(S, order.strike, tau, r, sigma, order.option_type)
            delta = bs_delta(S, order.strike, tau, r, sigma, order.option_type)
            gamma = bs_gamma(S, order.strike, tau, r, sigma)
            vega = bs_vega(S, order.strike, tau, r, sigma)
            theta = bs_theta(S, order.strike, tau, r, sigma, order.option_type)
            rho = bs_rho(S, order.strike, tau, r, sigma, order.option_type)

            records.append({
                "option_type": order.option_type,
                "strike": order.strike,
                "qty": order.quantity,
                "price": price * order.quantity,
                "delta": delta * order.quantity,
                "gamma": gamma * order.quantity,
                "vega": vega * order.quantity,
                "theta": theta * order.quantity,
                "rho": rho * order.quantity,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        totals = df[["price", "delta", "gamma", "vega", "theta", "rho"]].sum()
        totals["option_type"] = "TOTAL"
        totals["strike"] = "-"
        totals["qty"] = df["qty"].sum()
        df_tot = pd.concat([df, totals.to_frame().T], ignore_index=True)
        return df_tot

    # ------------------------------------------------------------------
    # Helpers pour grecs/PnL time series (pour les graphiques)
    # ------------------------------------------------------------------
    def greeks_over_time(self) -> pd.DataFrame:
        """Calcule la série temporelle agrégée des grecques du portefeuille."""
        coarse_len = len(self.path["coarse_idx"])
        rows = []
        for idx in range(coarse_len):
            st = self.state_at(idx)
            if st.empty:
                rows.append({"idx": idx, "price": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0})
            else:
                tot = st.iloc[-1]  # dernière ligne = TOTAL
                rows.append({"idx": idx, **tot[["price", "delta", "gamma", "vega", "theta", "rho"]]})
        return pd.DataFrame(rows)
