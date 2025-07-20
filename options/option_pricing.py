import numpy as np
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcul de d1 pour Black-Scholes.
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calcul de d2 pour Black-Scholes.
    """
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
    Prix BS d’une option européenne (annualisé).
    """
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
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
    Delta BS.
    """
    D1 = d1(S, K, T, r, sigma)
    return norm.cdf(D1) if option_type.lower() == "call" else norm.cdf(D1) - 1


def bs_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Gamma BS.
    """
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
    Vega BS (variation du prix par variation de sigma).
    """
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
    Theta BS annualisé (dérivée par rapport au temps).
    """
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(D1)
    # Terme de diffusion
    diffusion = - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    # Terme de financement
    if option_type.lower() == "call":
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
    Rho BS annualisé (dérivée par rapport au taux d'intérêt).
    """
    D2 = d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    else:
        return - K * T * np.exp(-r * T) * norm.cdf(-D2)
