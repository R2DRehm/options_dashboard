import numpy as np

def generate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int = None
) -> np.ndarray:
    """
    Génère M trajectoires de prix selon un modèle de Geometric Brownian Motion.
    
    Params:
    - S0: prix initial
    - mu: drift annualisé (rendement espéré)
    - sigma: volatilité annualisée
    - T: horizon en années (ex: 1 = 1 an)
    - N: nombre de pas de temps
    - M: nombre de trajectoires
    - seed: graine aléatoire (optionnel)
    
    Retour:
    - np.ndarray de forme (M, N+1) contenant les prix simulés
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    # incréments de Wiener
    dW = np.random.normal(scale=np.sqrt(dt), size=(M, N))
    # exponentiel du GBM
    increments = (mu - 0.5 * sigma**2) * dt + sigma * dW
    log_paths = np.cumsum(increments, axis=1)
    # Ajout de S0 au début
    S_paths = S0 * np.exp(np.hstack([np.zeros((M, 1)), log_paths]))
    return S_paths
