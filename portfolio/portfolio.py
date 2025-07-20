import numpy as np
import pandas as pd
from simulation.price_simulation import generate_gbm_paths
from simulation.vol_rates_simulation import generate_ou_paths, generate_vasicek_paths
from options.option_pricing import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho
)


class OptionPosition:
    def __init__(
        self,
        option_type: str,
        K: float,
        T0: float,
        quantity: float = 1.0
    ):
        """
        option_type: 'call' ou 'put'
        K: strike
        T0: maturité initiale (années)
        quantity: nombre de contrats (>0 long, <0 short)
        """
        self.option_type = option_type.lower()
        self.K = K
        self.T0 = T0
        self.quantity = quantity

    def pnl_and_greeks(
        self,
        S_path: np.ndarray,
        r_path: np.ndarray,
        sigma_path: np.ndarray,
        time_grid: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcule PnL et grecs le long d'une trajectoire:
        - S_path: prix spot (taille N+1)
        - r_path: taux sans risque (taille N+1)
        - sigma_path: volatilité spot (taille N+1)
        - time_grid: instants t de 0 à T0 (taille N+1)
        Retourne DataFrame ['t','S','r','sigma','price','delta','gamma','vega','theta','rho','pnl']
        """
        records = []
        # Prix initial
        price0 = bs_price(S_path[0], self.K, self.T0, r_path[0], sigma_path[0], self.option_type)
        for t, S, r, vol in zip(time_grid, S_path, r_path, sigma_path):
            tau = max(self.T0 - t, 0)
            price = bs_price(S, self.K, tau, r, vol, self.option_type)
            delta = bs_delta(S, self.K, tau, r, vol, self.option_type)
            gamma = bs_gamma(S, self.K, tau, r, vol)
            vega  = bs_vega(S, self.K, tau, r, vol)
            theta = bs_theta(S, self.K, tau, r, vol, self.option_type)
            rho   = bs_rho(S, self.K, tau, r, vol, self.option_type)
            pnl   = (price - price0) * self.quantity
            records.append({
                't': t,
                'S': S,
                'r': r,
                'sigma': vol,
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'rho': rho,
                'pnl': pnl
            })
        return pd.DataFrame(records)


class PortfolioSimulator:
    def __init__(
        self,
        positions: list[OptionPosition],
        S_params: dict,
        vol_params: dict,
        rate_params: dict,
        T: float,
        N: int,
        M: int,
        seed: int = None
    ):
        """
        positions: liste d'OptionPosition
        S_params: dict{'S0','mu','sigma'} pour GBM
        vol_params: dict{'x0','kappa','theta','xi'} pour OU
        rate_params: dict{'r0','kappa','theta','sigma'} pour Vasicek
        T: horizon (années), N: pas de temps, M: nb trajectoires
        seed: graine unifiée
        """
        self.positions = positions
        self.S_params = S_params
        self.vol_params = vol_params
        self.rate_params = rate_params
        self.T = T
        self.N = N
        self.M = M
        self.seed = seed

    def run(self) -> pd.DataFrame:
        """
        Simule trajectoires et calcule PnL & grecques pour chaque position et scénario.
        Retourne DataFrame concaténé avec colonnes ['scenario','position','t',...].
        """
        # Dérouler paramètres
        S0, mu, sigma_sim = self.S_params['S0'], self.S_params['mu'], self.S_params['sigma']
        x0, kappa_v, theta_v, xi = (
            self.vol_params['x0'], self.vol_params['kappa'],
            self.vol_params['theta'], self.vol_params['xi']
        )
        r0, kappa_r, theta_r, sigma_r = (
            self.rate_params['r0'], self.rate_params['kappa'],
            self.rate_params['theta'], self.rate_params['sigma']
        )
        # Simulations
        price_paths = generate_gbm_paths(S0, mu, sigma_sim, self.T, self.N, self.M, seed=self.seed)
        vol_paths   = generate_ou_paths(x0, kappa_v, theta_v, xi, self.T, self.N, self.M, seed=self.seed)
        rate_paths  = generate_vasicek_paths(r0, kappa_r, theta_r, sigma_r, self.T, self.N, self.M, seed=self.seed)
        time_grid = np.linspace(0, self.T, self.N + 1)

        results = []
        for i in range(self.M):
            for pos in self.positions:
                df = pos.pnl_and_greeks(
                    price_paths[i], rate_paths[i], vol_paths[i], time_grid
                )
                df['scenario'] = i
                df['position'] = f"{pos.option_type}_{pos.K}"
                results.append(df)
        return pd.concat(results, ignore_index=True)
