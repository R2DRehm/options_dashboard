import numpy as np
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcul de d1 pour Black-Scholes. Retourne NaN si T <= 0.
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcul de d2 pour Black-Scholes. Retourne NaN si T <= 0.
    """
    if T <= 0 or sigma <= 0:
        return np.nan
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Prix BS d’une option européenne. Pour T <= 0, renvoie le payoff.
    """
    option_type = option_type.lower()
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Delta BS. Pour T <= 0, renvoie 1 ou 0 pour call, -1 ou 0 pour put.
    """
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    D1 = d1(S, K, T, r, sigma)
    return norm.cdf(D1) if option_type == "call" else norm.cdf(D1) - 1.0


def bs_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Gamma BS. Renvoie 0 si T <= 0.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Vega BS. Renvoie 0 si T <= 0.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)


def bs_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Theta BS annualisé. Renvoie 0 si T <= 0.
    """
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(D1)
    diffusion = - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        finance = - r * K * np.exp(-r * T) * norm.cdf(D2)
    else:
        finance = r * K * np.exp(-r * T) * norm.cdf(-D2)
    return diffusion + finance


def bs_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Rho BS annualisé. Renvoie 0 si T <= 0.
    """
    option_type = option_type.lower()
    if T <= 0:
        return 0.0
    D2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    else:
        return - K * T * np.exp(-r * T) * norm.cdf(-D2)
