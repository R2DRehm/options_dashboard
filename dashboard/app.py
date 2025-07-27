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

def sidebar() -> html.Div:
    """Left panel: simulation params + order form + portfolio table."""
    return html.Div(
        style={"width": "25%", "padding": "15px",
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
    """Right panel: graphs in two columns + slider + PnL graph."""
    small_style = {"height": "230px"}

    # Grille 2 colonnes, 7 graphiques (Spot, PnL, puis 5 grecques)
    graphs_grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "10px",
        },
        children=[
            dcc.Graph(id="graph-paths",  style=small_style),
            dcc.Graph(id="graph-pnl",    style=small_style),
            dcc.Graph(id="graph-delta",  style=small_style),
            dcc.Graph(id="graph-gamma",  style=small_style),
            dcc.Graph(id="graph-vega",   style=small_style),
            dcc.Graph(id="graph-theta",  style=small_style),
            dcc.Graph(id="graph-rho",    style=small_style),
        ],
    )

    return html.Div(
        style={"width": "75%", "padding": "15px"},
        children=[
            graphs_grid,
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
    ],
)

# ----------------------------------------------------------------------------
# Callbacks – SIMULATE (regeneration) ----------------------------------------
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Callbacks – ORDERS (add / remove / reset) ----------------------------------
# ----------------------------------------------------------------------------
@app.callback(
    Output("orders-json", "data", allow_duplicate=True),
    Input("btn-add", "n_clicks"),
    State("orders-json", "data"),
    State("time-slider", "value"),
    State("instr-type", "value"),  
    State("strike", "value"),
    State("qty", "value"),
    State("maturity", "value"),
    prevent_initial_call=True,
)
def add_order_callback(_, orders, idx, instr, strike, qty, maturity):
    if orders is None:
        orders = []
    orders.append({
        "time_idx": int(idx),
        "option_type": instr,    
        "strike": strike or 0.0,
        "quantity": qty,
        "maturity": maturity or 0.0
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
    Output("graph-paths", "figure"),
    Output("graph-pnl",   "figure"),
    Output("graph-delta", "figure"),
    Output("graph-gamma", "figure"),
    Output("graph-vega",  "figure"),
    Output("graph-theta","figure"),
    Output("graph-rho",   "figure"),
    Output("table-positions", "data"),
    Output("table-positions", "columns"),
    Input("time-slider", "value"),
    Input("paths-json",  "data"),
    Input("orders-json", "data"),
    prevent_initial_call=True,
)

def update_view(idx: int, paths_json: dict | None, orders: list | None):
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
    fig_paths.update_layout(title="Spot / σ / r – full path", height=230,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # ---------- 3) Construire le portefeuille ----------
    portfolio = Portfolio(path)
    for od in orders or []:
        portfolio.add_order(OptionOrder(**od))

    greeks_df = portfolio.greeks_over_time()

    # ----- PnL over time -----
    pnl_df = greeks_df[['idx','price']].rename(columns={'price':'pnl'})
    fig_pnl = px.line(pnl_df, x='idx', y='pnl', title='Portfolio PnL vs snapshot')
    fig_pnl.add_vline(x=idx, line_width=1, line_dash='dot', line_color='black')
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
    snapshot_df = portfolio.state_at(idx)
    if snapshot_df.empty:
        return fig_paths, fig_pnl, fig_delta, fig_gamma, fig_vega, fig_theta, fig_rho, [], []

    table_data = snapshot_df.to_dict("records")
    columns    = [{"name": c, "id": c} for c in snapshot_df.columns]
    return fig_paths, fig_pnl, fig_delta, fig_gamma, fig_vega, fig_theta, fig_rho, table_data, columns


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
