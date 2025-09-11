"""dashboard/app.py – Option Trading Simulator

- Simulation corrélée Spot / Vol (variance) et taux (indépendant)
- Slider (N_coarse points) pour naviguer dans le temps
- Sidebar : paramètres de simulation + formulaire d’ordre (call/put, strike, qty, maturité)
- Graphiques : trajectoire complète S, σ, r ; grecques portefeuille (Δ, Γ, Vega, Theta, Rho)
- Tableau : positions actives + TOTAL au snapshot sélectionné
- Boutons : Simulate (régénère une trajectoire) ; Add Order ; Remove Last (facile) ; Reset Orders

Tous les états sont stockés via dcc.Store afin que l’interface reste réactive.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from simulation.correlated_paths import generate_paths, PathParams
from portfolio.portfolio import Portfolio, OptionOrder

# ----------------------------------------------------------------------------
# Dash app initialisation
# ----------------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ----------------------------------------------------------------------------
# Layout helpers
# ----------------------------------------------------------------------------

from dash import dcc, html, dash_table   # vérifie que dash_table est bien importé

import math

def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(is_call: bool, S: float, K: float, r: float, sigma: float, tau: float) -> float:
    """Black–Scholes européen; gère tau<=0 et sigma<=0 (valeur intrinsèque)."""
    tau = max(tau, 0.0)
    if tau == 0.0 or sigma <= 0.0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    if is_call:
        return S * _N(d1) - K * math.exp(-r * tau) * _N(d2)
    else:
        return K * math.exp(-r * tau) * _N(-d2) - S * _N(-d1)

def price_of_order_at_idx(path: dict, od: dict, idx_snap: int) -> float:
    """Prix (par unité) d’un ordre au snapshot idx_snap (0..Nc-1) à partir de path."""
    j = int(path["coarse_idx"][idx_snap])
    t = float(path["t_fine"][j])
    S = float(path["S"][j])
    r = float(path["r"][j])
    sigma = float(np.sqrt(path["V"][j]))
    typ = od["option_type"]
    if typ == "underlying":
        return S
    elif typ in ("call", "put"):
        tau = max((od.get("maturity") or 0.0) - t, 0.0)
        return bs_price(typ == "call", S, float(od.get("strike") or 0.0), r, sigma, tau)
    else:
        return 0.0


def sidebar() -> html.Div:
    """Left panel: simulation params + order form + portfolio table."""
    return html.Div(
        style={"width": "20%", "padding": "15px",
               "overflowY": "auto", "background": "#f1f3f4"},
        children=[
            html.H3("Simulation"),
            html.Label("S0"),
            dcc.Input(id="s0", type="number", value=100, step=1),

            html.Label("V0 (σ²)"),
            dcc.Input(id="v0", type="number", value=0.04, step=0.01),

            html.Label("r0"),
            dcc.Input(id="r0", type="number", value=0.01, step=0.001),
            html.Br(),

            html.Label("μ"),
            dcc.Input(id="mu", type="number", value=0.05, step=0.01),

            html.Label("κ_v"),
            dcc.Input(id="kappa_v", type="number", value=1.0, step=0.1),

            html.Label("θ_v"),
            dcc.Input(id="theta_v", type="number", value=0.04, step=0.01),

            html.Label("ξ"),
            dcc.Input(id="xi", type="number", value=0.2, step=0.01),
            html.Br(),

            html.Label("κ_r"),
            dcc.Input(id="kappa_r", type="number", value=0.5, step=0.1),

            html.Label("θ_r"),
            dcc.Input(id="theta_r", type="number", value=0.03, step=0.01),

            html.Label("σ_r"),
            dcc.Input(id="sigma_r", type="number", value=0.01, step=0.001),
            html.Br(),

            html.Label("ρ (corr S-σ)"),
            dcc.Input(id="rho", type="number", value=-0.5, step=0.05),
            html.Br(),

            html.Label("T (years)"),
            dcc.Input(id="T", type="number", value=1.0, step=0.1),

            html.Label("Fine steps"),
            dcc.Input(id="Nf", type="number", value=2000, step=100),

            html.Label("Coarse pts"),
            dcc.Input(id="Nc", type="number", value=10, step=1),
            html.Br(),

            html.Label("Seed"),
            dcc.Input(id="seed", type="number", value=42, step=1),
            html.Br(),

            html.Button("Simulate", id="btn-sim", n_clicks=0,
                        style={"marginTop": "10px", "width": "100%"}),

            html.Hr(),  # ------------- ORDERS -------------
            html.H3("Add / Remove position"),

            # ─── Instrument selector ──────────────────────────
            html.Label("Instrument"),
            dcc.Dropdown(
                id="instr-type",
                options=[
                    {"label": "Underlying", "value": "underlying"},
                    {"label": "Call",       "value": "call"},
                    {"label": "Put",        "value": "put"},
                ],
                value="underlying",
            ),

            html.Label("Strike"),
            dcc.Input(id="strike", type="number", value=100, step=1),

            html.Label("Qty (+ long / − short)"),
            dcc.Input(id="qty", type="number", value=1, step=0.1),

            html.Label("Maturity (years)"),
            dcc.Input(id="maturity", type="number", value=1.0, step=0.1),
            html.Br(),

            html.Button("Add",   id="btn-add",    n_clicks=0, style={"width": "49%", "marginTop": "8px"}),
            html.Button("Remove last", id="btn-remove", n_clicks=0, style={"width": "49%", "marginLeft": "2%"}),
            html.Br(),
            html.Button("Reset", id="btn-reset", n_clicks=0, style={"marginTop": "10px", "width": "100%"}),


            html.Hr(),  # ------------- PORTFOLIO TABLE -------------
            html.H3("Portfolio @ snapshot"),
            dash_table.DataTable(
                id="table-positions",     # << important !
                page_size=8,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "right"},
            ),
        ],
    )



def main_panel() -> html.Div:
    """Right panel with left/right arrows and paged content."""
    nav = html.Div(
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "10px"},
        children=[
            html.Button("◀", id="btn-prev", n_clicks=0, style={"width": "60px"}),
            html.Button("▶", id="btn-next", n_clicks=0, style={"width": "60px"}),
        ],
    )

    # Container whose children we will swap (graphs or table)
    page = html.Div(id="page-content")

    return html.Div(
        style={"width": "80%", "padding": "15px"},
        children=[
            nav,
            page,
            html.Br(),
            dcc.Slider(id="time-slider", min=0, max=10, step=1, value=0),
        ],
    )




app.layout = html.Div(
    style={"display": "flex", "height": "100vh"},
    children=[
        sidebar(),
        main_panel(),
        # Hidden stores ----------------------------------------------
        dcc.Store(id="paths-json"),  # trajectoires simulées
        dcc.Store(id="orders-json", data=[]),  # ordre book
        dcc.Store(id="page-idx", data=0),
    ],
)

@app.callback(
    Output("page-idx", "data", allow_duplicate=True),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("page-idx", "data"),
    prevent_initial_call=True,
)
def turn_page(prev, nxt, idx):
    triggered = ctx.triggered_id
    if triggered == "btn-prev":
        idx = (idx - 1) % 3
    elif triggered == "btn-next":
        idx = (idx + 1) % 3
    return idx


@app.callback(
    Output("paths-json", "data"),
    Output("orders-json", "data"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Input("btn-sim", "n_clicks"),
    State("s0", "value"), State("v0", "value"), State("r0", "value"),
    State("mu", "value"), State("kappa_v", "value"), State("theta_v", "value"), State("xi", "value"),
    State("kappa_r", "value"), State("theta_r", "value"), State("sigma_r", "value"),
    State("rho", "value"), State("T", "value"), State("Nf", "value"), State("Nc", "value"), State("seed", "value"),
    prevent_initial_call=True,
)
def simulate_callback(n_clicks, S0, V0, r0, mu, kappa_v, theta_v, xi, kappa_r, theta_r, sigma_r, rho, T, Nf, Nc, seed):
    """Regenerate a new path and reset orders + slider."""
    params = PathParams(mu=mu, kappa_v=kappa_v, theta_v=theta_v, xi=xi, kappa_r=kappa_r, theta_r=theta_r, sigma_r=sigma_r, rho=rho)
    paths = generate_paths(S0, V0, r0, params, T, int(Nf), int(Nc), seed=int(seed) if seed is not None else None)
    # JSON-serialisable
    paths_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in paths.items()}
    return paths_json, [], int(Nc), 0

@app.callback(
    Output("orders-json", "data", allow_duplicate=True),
    Input("btn-add", "n_clicks"),
    State("orders-json", "data"),
    State("time-slider", "value"),
    State("instr-type", "value"),
    State("strike", "value"),
    State("qty", "value"),
    State("maturity", "value"),
    State("paths-json", "data"),         
    prevent_initial_call=True,
)
def add_order_callback(_, orders, idx, instr, strike, qty, maturity, paths_json):
    if orders is None:
        orders = []
    # reconstituer le path numpy
    path = {k: np.array(v) if isinstance(v, list) else v for k, v in (paths_json or {}).items()}
    # prix d’entrée au snapshot courant
    entry_price = price_of_order_at_idx(path, {
        "option_type": instr, "strike": strike or 0.0, "maturity": maturity or 0.0
    }, int(idx))

    orders.append({
        "time_idx": int(idx),
        "option_type": instr,
        "strike": strike or 0.0,
        "quantity": qty,
        "maturity": maturity or 0.0,
        "entry_price": float(entry_price),   # <<< stocké pour CASH
    })
    return orders




@app.callback(
    Output("orders-json", "data", allow_duplicate=True),
    Input("btn-remove", "n_clicks"),
    State("orders-json", "data"),
    State("time-slider", "value"),
    prevent_initial_call=True,
)
def remove_last_callback(_, orders, idx):
    if not orders:
        return orders
    # Remove last order whose time_idx <= current slider idx
    for i in range(len(orders) - 1, -1, -1):
        if orders[i]["time_idx"] <= idx:
            orders.pop(i)
            break
    return orders


@app.callback(
    Output("orders-json", "clear_data"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_orders_callback(_):
    return True

# ------------------------------------------------------------------
# UPDATE VIEW  (slider / paths / orders)
# ------------------------------------------------------------------
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


@app.callback(
    Output("page-content", "children"),
    Output("table-positions", "data"),
    Output("table-positions", "columns"),
    Input("time-slider", "value"),
    Input("paths-json",  "data"),
    Input("orders-json", "data"),
    Input("page-idx",    "data"),      # ← page-idx devient un INPUT
    prevent_initial_call=True,
)
def update_view(idx: int, paths_json: dict | None, orders: list | None, page_idx: int):
    if not paths_json:
        raise dash.exceptions.PreventUpdate

    # ---------- 1) Reconstituer la trajectoire ----------
    path = {k: np.array(v) if isinstance(v, list) else v for k, v in paths_json.items()}

    # Snapshot courant
    j      = int(path["coarse_idx"][idx])
    t_now  = path["t_fine"][j]

    # ---------- 2) Graphe Spot / σ / r (axes distincts) ----------
    fig_paths = make_subplots(specs=[[{"secondary_y": True}]])
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=path["S"],          name="Spot"), secondary_y=False)
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=np.sqrt(path["V"]), name="σ"),    secondary_y=True)
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=path["r"],          name="r"),    secondary_y=True)
    fig_paths.add_vline(x=t_now, line_width=1, line_dash="dot", line_color="black")
    fig_paths.update_layout(
        title="Spot / σ / r – full path",
        height=320,  # Hauteur alignée sur le CSS du main_panel()
    )


    # ---------- 3) Construire le portefeuille ----------
    portfolio = Portfolio(path)

    # Seules les clés acceptées par OptionOrder
    _ALLOWED = {"time_idx", "option_type", "strike", "quantity", "maturity"}

    for od in (orders or []):
        od_clean = {k: od.get(k) for k in _ALLOWED if k in od}
        # cast défensif (au cas où JSON renvoie None)
        od_clean["time_idx"] = int(od_clean.get("time_idx", 0))
        od_clean["strike"]   = float(od_clean.get("strike", 0.0) or 0.0)
        od_clean["quantity"] = float(od_clean.get("quantity", 0.0) or 0.0)
        od_clean["maturity"] = float(od_clean.get("maturity", 0.0) or 0.0)
        portfolio.add_order(OptionOrder(**od_clean))


    greeks_df = portfolio.greeks_over_time()

    # ----- Equity / Cash / MtM / PnL over time -----
    Nc = len(path["coarse_idx"])
    idxs = np.arange(Nc)

    # MtM(t): somme des valeurs courantes des positions
    mtm = np.zeros(Nc)
    for k in range(Nc):
        v = 0.0
        for od in (orders or []):
            v += (float(od["quantity"]) * price_of_order_at_idx(path, od, k))
        mtm[k] = v

    # Cash(t): somme des cashflows de trades déjà passés (prix d’entrée * quantité, signe d’achat/vente)
    # Convention: achat qty>0 => cash sortant => - qty * entry_price
    cash = np.zeros(Nc)
    if orders:
        # cashflow par ordre à la date d’exécution
        cf = np.zeros(Nc)
        for od in orders:
            k0 = int(od["time_idx"])
            ep = float(od.get("entry_price") or price_of_order_at_idx(path, od, k0))
            cf[k0] += - float(od["quantity"]) * ep
        cash = np.cumsum(cf)

    equity = mtm + cash
    pnl_series = equity  # Equity(0)=0 ici; sinon soustraire equity[0]
    pnl_df = pd.DataFrame({"idx": idxs, "pnl": pnl_series})

    fig_pnl = px.line(pnl_df, x="idx", y="pnl", title="Portfolio PnL vs snapshot")
    fig_pnl.add_vline(x=idx, line_width=1, line_dash="dot", line_color="black")
    fig_pnl.update_layout(height=230, margin=dict(l=50, r=20, t=30, b=20))


    # Helper pour chaque grecque
    def greek_fig(column: str, title: str):
        fig = px.line(greeks_df, x="idx", y=column, title=title)
        fig.add_vline(x=idx, line_width=1, line_dash="dot", line_color="black")
        fig.update_layout(height=230, margin=dict(l=50, r=20, t=30, b=20))
        return fig

    fig_delta = greek_fig("delta",  "Δ  (Delta)")
    fig_gamma = greek_fig("gamma",  "Γ  (Gamma)")
    fig_vega  = greek_fig("vega",   "Vega")
    fig_theta = greek_fig("theta",  "Theta")
    fig_rho   = greek_fig("rho",    "Rho")

    # ---------- 4) Tableau des positions ----------
    rows = []
    for n, od in enumerate(orders or []):
        entry_idx = int(od["time_idx"])
        entry_price = float(od.get("entry_price") or price_of_order_at_idx(path, od, entry_idx))
        cur_price = float(price_of_order_at_idx(path, od, idx))
        qty = float(od["quantity"])
        rows.append({
            "id": n,
            "type": od["option_type"],
            "qty": qty,
            "strike": od.get("strike", ""),
            "maturity": od.get("maturity", ""),
            "entry_idx": entry_idx,
            "entry_price": round(entry_price, 6),
            "mtm_price": round(cur_price, 6),
            "value": round(qty * cur_price, 6),
            "uPnL": round(qty * (cur_price - entry_price), 6),
        })
    snapshot_df = pd.DataFrame(rows)
    table_data = snapshot_df.to_dict("records")
    columns = [{"name": c, "id": c} for c in snapshot_df.columns]

    # ---------- 5) Choix du contenu de page ----------
    if page_idx == 0:
        content = dcc.Graph(figure=fig_paths, style={"height": "500px"})
    elif page_idx == 1:
        greeks_grid = html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
            children=[
                dcc.Graph(figure=fig_pnl,   style={"height": "230px"}),
                dcc.Graph(figure=fig_delta, style={"height": "230px"}),
                dcc.Graph(figure=fig_gamma, style={"height": "230px"}),
                dcc.Graph(figure=fig_vega,  style={"height": "230px"}),
                dcc.Graph(figure=fig_theta, style={"height": "230px"}),
                dcc.Graph(figure=fig_rho,   style={"height": "230px"}),
            ],
        )
        content = greeks_grid
    else:  # page 2
        content = dash_table.DataTable(
            data=table_data,
            columns=columns,
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "right"},
        )

    return content, table_data, columns




# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
