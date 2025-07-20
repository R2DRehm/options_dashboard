# utils/helpers.py - updated to avoid FutureWarnings and use groupby['pnl'].last()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def plot_price_paths(df: pd.DataFrame, scenario: int = None):
    """
    Trace les trajectoires de prix S sur le temps pour un ou plusieurs scénarios.
    - df doit contenir les colonnes ['t','S','scenario'].
    - scenario: si spécifié, ne trace que ce scénario.
    """
    data = df.copy()
    if scenario is not None:
        data = data[data['scenario'] == scenario]
    fig = px.line(data, x='t', y='S', color='scenario', title='Trajectoires de prix')
    fig.show()
    return fig


def plot_greek_paths(df: pd.DataFrame, greek: str, scenario: int = None):
    """
    Trace l'évolution d'un grec donné ('delta','gamma','vega','theta','rho').
    - df doit contenir ['t', greek, 'scenario'].
    - scenario: si spécifié, filtre sur ce scénario.
    """
    data = df.copy()
    if scenario is not None:
        data = data[data['scenario'] == scenario]
    fig = px.line(data, x='t', y=greek, color='scenario', title=f'Évolution de {greek}')
    fig.show()
    return fig


def plot_pnl_paths(df: pd.DataFrame, scenario: int = None):
    """
    Trace le PnL au fil du temps.
    - df doit contenir ['t','pnl','scenario'].
    """
    data = df.copy()
    if scenario is not None:
        data = data[data['scenario'] == scenario]
    fig = px.line(data, x='t', y='pnl', color='scenario', title='PnL le long de la trajectoire')
    fig.show()
    return fig


def summarize_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des statistiques de PnL final par scénario :
    - mean, std, median, quantiles (5%, 95%)
    """
    last = df.groupby('scenario')['pnl'].last()
    summary = pd.DataFrame({
        'mean': last.mean(),
        'std': last.std(),
        'median': last.median(),
        '5%': last.quantile(0.05),
        '95%': last.quantile(0.95)
    }, index=['value']).T
    return summary


def compute_var(df: pd.DataFrame, alpha: float = 0.05) -> float:
    """
    Calcule la VaR à niveau alpha sur le PnL final de tous les scénarios.
    VaR est le quantile alpha (perte maximale) :
    """
    last = df.groupby('scenario')['pnl'].last()
    return np.percentile(-last.values, 100 * alpha)


def save_simulation(df: pd.DataFrame, filepath: str):
    """
    Sauvegarde le DataFrame de simulation au format CSV.
    """
    df.to_csv(filepath, index=False)


def load_simulation(filepath: str) -> pd.DataFrame:
    """
    Charge un CSV de simulation et renvoie un DataFrame.
    """
    return pd.read_csv(filepath)
