import numpy as np


def generate_ou_paths(
    x0: float,
    kappa: float,
    theta: float,
    xi: float,
    T: float,
    N: int,
    M: int,
    seed: int = None
) -> np.ndarray:
    """
    Simule M trajectoires d'un processus Ornstein-Uhlenbeck:
        dx_t = kappa*(theta - x_t) dt + xi dW_t
    utile pour modéliser la volatilité (OU en valeurs absolues).

    Params:
    - x0: valeur initiale
    - kappa: vitesse de réversion
    - theta: niveau moyen
    - xi: volatilité du processus OU
    - T: horizon en années
    - N: nombre de pas de temps
    - M: nombre de trajectoires
    - seed: graine pour la reproductibilité

    Retour:
    - np.ndarray de forme (M, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = x0

    for t in range(N):
        dw = np.random.normal(scale=np.sqrt(dt), size=M)
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (theta - paths[:, t]) * dt
            + xi * dw
        )
    return paths


def generate_vasicek_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int = None
) -> np.ndarray:
    """
    Simule M trajectoires du modèle de Vasicek pour les taux d'intérêt:
        dr_t = kappa*(theta - r_t) dt + sigma dW_t

    Params:
    - r0: taux initial
    - kappa: vitesse de réversion
    - theta: taux moyen vers lequel le processus revient
    - sigma: volatilité des taux
    - T: horizon en années
    - N: nombre de pas de temps
    - M: nombre de trajectoires
    - seed: graine pour la reproductibilité

    Retour:
    - np.ndarray de forme (M, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = r0

    for t in range(N):
        dw = np.random.normal(scale=np.sqrt(dt), size=M)
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (theta - paths[:, t]) * dt
            + sigma * dw
        )
    return paths
