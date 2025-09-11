"""portfolio/portfolio.py – Gestion dynamique de portefeuille d’options

Principales classes :
    • OptionOrder   – décrit un ordre passé sur le grid discret.
    • Portfolio     – agrège les ordres et calcule PnL & grecques au snapshot voulu.

Dépendances :
    - simulation.correlated_paths.generate_paths
    - options.option_pricing (grecs, pricing)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd



def _maturity_remaining(T0: float, t_current: float) -> float:
    """Maturité résiduelle non‑négative."""
    return max(T0 - t_current, 0.0)


@dataclass
class OptionOrder:
    """Ordre exécuté au snapshot time_idx."""
    time_idx: int                  # indice sur le coarse grid
    option_type: str               # "underlying" | "call" | "put"
    strike: float
    quantity: float                # + long / − short
    maturity: float                # maturité absolue (années depuis t=0)
    entry_price: float | None = None  # prix par unité au moment de l’exécution



class Portfolio:
    """Portefeuille d’OptionOrder sur une trajectoire simulée."""

    def __init__(self, path_dict: dict[str, np.ndarray]):
        self.path = path_dict  # dict retourné par generate_paths
        self.orders: List[OptionOrder] = []

    def _snapshot_params(self, idx: int):
        j = int(self.path["coarse_idx"][idx])
        t = float(self.path["t_fine"][j])
        S = float(self.path["S"][j])
        r = float(self.path["r"][j])
        sigma = float(np.sqrt(self.path["V"][j]))
        return j, t, S, r, sigma

    def _tau(self, T0: float, t_now: float) -> float:
        return max(T0 - t_now, 0.0)

    # ------------------------------------------------------------------
    # Gestion des ordres
    # ------------------------------------------------------------------
    def add_order(self, order: OptionOrder):
        self.orders.append(order)

    # ------------------------------------------------------------------
    # Valeur + grecques au snapshot courant
    # ------------------------------------------------------------------
    def state_at(self, idx: int) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame()

        _, t_now, S, r, sigma = self._snapshot_params(idx)
        rows = []

        for n, od in enumerate(self.orders):
            if od.time_idx > idx:
                continue

            # prix et grecques "par unité"
            if od.option_type == "underlying":
                mtm_unit = S
                greeks_unit = {"delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
            else:
                tau = self._tau(od.maturity, t_now)
                from options.option_pricing import instrument_price_and_greeks
                pg = instrument_price_and_greeks(od.option_type, S, od.strike, tau, r, sigma)
                mtm_unit = float(pg["price"])
                greeks_unit = {k: float(pg[k]) for k in ["delta", "gamma", "vega", "theta", "rho"]}

            entry_price = od.entry_price
            # fallback si pas stocké : valorise au snapshot d’exécution
            if entry_price is None:
                _, t_exec, Sx, rx, sigx = self._snapshot_params(od.time_idx)
                if od.option_type == "underlying":
                    entry_price = Sx
                else:
                    from options.option_pricing import instrument_price
                    entry_price = float(instrument_price(od.option_type, Sx, od.strike, self._tau(od.maturity, t_exec), rx, sigx))

            qty = float(od.quantity)
            cur_value = qty * mtm_unit
            uPnL = qty * (mtm_unit - float(entry_price))

            rows.append({
                "id": n,
                "option_type": od.option_type,
                "strike": od.strike if od.option_type != "underlying" else "-",
                "qty": qty,
                "entry_idx": od.time_idx,
                "entry_price": entry_price,
                "mtm_price": mtm_unit,
                "value": cur_value,
                "uPnL": uPnL,
                "delta": qty * greeks_unit["delta"],
                "gamma": qty * greeks_unit["gamma"],
                "vega":  qty * greeks_unit["vega"],
                "theta": qty * greeks_unit["theta"],
                "rho":   qty * greeks_unit["rho"],
                "maturity": od.maturity if od.option_type != "underlying" else "-",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        totals = df[["value","uPnL","delta","gamma","vega","theta","rho"]].sum()
        totals["id"] = "TOTAL"
        totals["option_type"] = "TOTAL"
        totals["strike"] = "-"
        totals["qty"] = df["qty"].sum()
        totals["entry_idx"] = ""
        totals["entry_price"] = ""
        totals["mtm_price"] = ""
        totals["maturity"] = ""
        df_tot = pd.concat([df, totals.to_frame().T], ignore_index=True)

        # Ordonner les colonnes pour l’affichage
        col_order = ["id","option_type","qty","strike","maturity","entry_idx",
                    "entry_price","mtm_price","value","uPnL",
                    "delta","gamma","vega","theta","rho"]
        return df_tot[col_order]


    # ------------------------------------------------------------------
    # Helpers pour grecs/PnL time series (pour les graphiques)
    # ------------------------------------------------------------------

    def cash_series(self) -> np.ndarray:
        """Cash(t) = cumul des flux d'exécution (− qty * entry_price) au fil des snapshots."""
        Nc = len(self.path["coarse_idx"])
        cf = np.zeros(Nc)
        if not self.orders:
            return cf
        for od in self.orders:
            k0 = int(od.time_idx)
            # entry_price fallback si absent
            ep = od.entry_price
            if ep is None:
                _, t_exec, Sx, rx, sigx = self._snapshot_params(k0)
                if od.option_type == "underlying":
                    ep = Sx
                else:
                    from options.option_pricing import instrument_price
                    ep = float(instrument_price(od.option_type, Sx, od.strike, self._tau(od.maturity, t_exec), rx, sigx))
            cf[k0] += - float(od.quantity) * float(ep)
        return np.cumsum(cf)

    def mtm_series(self) -> np.ndarray:
        """MtM agrégé du portefeuille sur les snapshots."""
        Nc = len(self.path["coarse_idx"])
        mtm = np.zeros(Nc)
        for k in range(Nc):
            st = self.state_at(k)
            if st.empty:
                continue
            mtm[k] = float(st.iloc[-1]["value"])  # TOTAL.value
        return mtm

    def pnl_over_time(self) -> pd.DataFrame:
        """DataFrame idx, cash, mtm, equity, pnl."""
        idxs = np.arange(len(self.path["coarse_idx"]))
        cash = self.cash_series()
        mtm  = self.mtm_series()
        equity = cash + mtm
        pnl = equity - equity[0]  # Equity(0)=0 en général
        return pd.DataFrame({"idx": idxs, "cash": cash, "mtm": mtm, "equity": equity, "pnl": pnl})

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
