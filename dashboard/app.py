"""dashboard/app.py – Option Trading Simulator

- Simulation corrélée Spot / Vol (variance) et taux (indépendant)
- Slider (N_coarse points) pour naviguer dans le temps
- Sidebar : paramètres de simulation + formulaire d’ordre (call/put, strike, qty, maturité)
- Graphiques : trajectoire complète S, σ, r ; grecques portefeuille (Δ, Γ, Vega, Theta, Rho)
- Tableau : positions actives + TOTAL au snapshot sélectionné
- Boutons : Simulate (régénère une trajectoire) ; Add Order ; Remove Last (facile) ; Reset Orders

Tous les états sont stockés via dcc.Store afin que l’interface reste réactive.
"""
from __future__ import annotations  # noqa: F404

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx


from simulation.vol_rates_simulation import generate_paths_heston2f, generate_paths_garch
from portfolio.portfolio import Portfolio, OptionOrder
from options.option_pricing import instrument_price 

# ----------------------------------------------------------------------------
# Dash app initialisation
# ----------------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ----------------------------------------------------------------------------
# Layout helpers
# ----------------------------------------------------------------------------


import math  # noqa: E402

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
            html.Br(),

            html.Label("V0 (σ²)"),
            dcc.Input(id="v0", type="number", value=0.04, step=0.01),
            html.Br(),

            html.Label("r0"),
            dcc.Input(id="r0", type="number", value=0.01, step=0.001),
            html.Br(),

            html.Label("μ"),
            dcc.Input(id="mu", type="number", value=0.05, step=0.01),
            html.Br(),

            html.Label("κ_v"),
            dcc.Input(id="kappa_v", type="number", value=1.0, step=0.1),
            html.Br(),

            html.Label("θ_v"),
            dcc.Input(id="theta_v", type="number", value=0.04, step=0.01),
            html.Br(),

            html.Label("ξ"),
            dcc.Input(id="xi", type="number", value=0.2, step=0.01),
            html.Br(),

            html.Label("κ_r"),
            dcc.Input(id="kappa_r", type="number", value=0.5, step=0.1),
            html.Br(),

            html.Label("θ_r"),
            dcc.Input(id="theta_r", type="number", value=0.03, step=0.01),
            html.Br(),

            html.Label("σ_r"),
            dcc.Input(id="sigma_r", type="number", value=0.01, step=0.001),
            html.Br(),

            html.Label("ρ (corr S-σ)"),
            dcc.Input(id="rho", type="number", value=-0.5, step=0.05),
            html.Br(),

            html.Label("T (years)"),
            dcc.Input(id="T", type="number", value=1.0, step=0.1),
            html.Br(),

            html.Label("Fine steps"),
            dcc.Input(id="Nf", type="number", value=2000, step=100),
            html.Br(),

            html.Label("Coarse pts"),
            dcc.Input(id="Nc", type="number", value=100, step=1),
            html.Br(),

            html.Label("Seed"),
            dcc.Input(id="seed", type="number", placeholder="(random)", debounce=True),
            html.Br(),

            html.Label("Vol model"),
            dcc.Dropdown(
                id="vol-model",
                options=[
                    {"label": "Heston 1-facteur (actuel)", "value": "h1f"},
                    {"label": "Heston 2-facteurs (clusters)", "value": "h2f"},
                    {"label": "GARCH(1,1) (fort clustering)", "value": "garch"},
                ],
                value="h2f",
            ),
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
            html.Br(),

            html.Label("Strike"),
            dcc.Input(id="strike", type="number", value=100, step=1),
            html.Br(),

            html.Label("Qty (+ long / − short)"),
            dcc.Input(id="qty", type="number", value=1, step=0.01),
            html.Br(),

            html.Label("Maturity (years)"),
            dcc.Input(id="maturity", type="number", value=1.0, step=0.01),
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

    slider = html.Div(
        id="time-slider-wrapper",
        children=[dcc.Slider(id="time-slider", min=0, max=10, step=1, value=0)]
    )

    return html.Div(
        style={"width": "80%", "padding": "15px"},
        children=[
            nav,
            page,
            html.Br(),
            slider,   # ← au lieu de mettre directement le Slider
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
        idx = (idx - 1) % 2   # ← 2 pages
    elif triggered == "btn-next":
        idx = (idx + 1) % 2
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
    State("rho", "value"), State("T", "value"), State("Nf", "value"), State("Nc", "value"), State("seed", "value"), State("vol-model","value"),
    prevent_initial_call=True,
)
def simulate_callback(n_clicks, S0, V0, r0, mu, kappa_v, theta_v, xi, kappa_r, theta_r, sigma_r, rho, T, Nf, Nc, seed, vol_model):
    params = PathParams(mu=mu, kappa_v=kappa_v, theta_v=theta_v, xi=xi,
                        kappa_r=kappa_r, theta_r=theta_r, sigma_r=sigma_r, rho=rho)

    seed_val = None if seed in (None, "") else int(seed)

    if vol_model == "h2f":
        paths = generate_paths_heston2f(S0, V0, r0, params, float(T), int(Nf), int(Nc), seed=seed_val)
    elif vol_model == "garch":
        paths = generate_paths_garch(S0, V0, r0, params, float(T), int(Nf), int(Nc), seed=seed_val)

    paths_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in paths.items()}
    return paths_json, [], int(Nc) - 1, 0


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
    path = {k: np.array(v) if isinstance(v, list) else v for k, v in (paths_json or {}).items()}
    j = int(path["coarse_idx"][int(idx)])
    t = float(path["t_fine"][j])
    S = float(path["S"][j])
    r = float(path["r"][j])
    sigma = float(np.sqrt(path["V"][j]))
    tau = max((maturity or 0.0) - t, 0.0)
    entry_price = S if instr == "underlying" else float(instrument_price(instr, S, float(strike or 0.0), tau, r, sigma))

    # --- dans add_order_callback()
    orders.append({
        "time_idx": int(idx),
        "option_type": instr,
        "strike": float(strike or 0.0),
        "quantity": round(float(qty or 0.0), 2),         
        "maturity": float(maturity or 0.0),
        "entry_price": float(entry_price),
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

@app.callback(
    Output("page-content", "children"),
    Output("table-positions", "data"),
    Output("table-positions", "columns"),
    Output("time-slider-wrapper", "style"),
    Input("time-slider", "value"),
    Input("paths-json",  "data"),
    Input("orders-json", "data"),
    Input("page-idx",    "data"),
    prevent_initial_call=True,
)
def update_view(idx: int, paths_json: dict | None, orders: list | None, page_idx: int):
    if not paths_json:
        raise dash.exceptions.PreventUpdate

    # ---------- 1) Path ----------
    path = {k: np.array(v) if isinstance(v, list) else v for k, v in paths_json.items()}
    j     = int(path["coarse_idx"][idx])
    t_now = float(path["t_fine"][j])

    # ---------- 2) Trajectoires S / σ / r ----------
    fig_paths = make_subplots(specs=[[{"secondary_y": True}]])
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=path["S"],          name="Spot"), secondary_y=False)
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=np.sqrt(path["V"]), name="σ"),    secondary_y=True)
    fig_paths.add_trace(go.Scatter(x=path["t_fine"], y=path["r"],          name="r"),    secondary_y=True)
    fig_paths.add_vline(x=t_now, line_width=1, line_dash="dot", line_color="black")
    fig_paths.update_layout(title="Spot / σ / r – full path", autosize=True, margin=dict(l=50, r=20, t=30, b=20))

    # ---------- 3) Portefeuille ----------
    portfolio = Portfolio(path)
    for od in (orders or []):
        portfolio.add_order(OptionOrder(
            time_idx=int(od.get("time_idx", 0)),
            option_type=od.get("option_type", "call"),
            strike=float(od.get("strike", 0.0) or 0.0),
            quantity=float(od.get("quantity", 0.0) or 0.0),
            maturity=float(od.get("maturity", 0.0) or 0.0),
            entry_price=(float(od["entry_price"]) if od.get("entry_price") is not None else None),
        ))

    greeks_df = portfolio.greeks_over_time()

    # ----- PnL over time -----
    pnl_df = portfolio.pnl_over_time()
    def _greek_fig(column: str, title: str):
        f = px.line(greeks_df, x="idx", y=column, title=title)
        f.add_vline(x=idx, line_width=1, line_dash="dot", line_color="black")
        f.update_layout(autosize=True, margin=dict(l=50, r=20, t=30, b=20))
        return f

    fig_pnl   = px.line(pnl_df, x="idx", y="pnl", title="Portfolio PnL vs snapshot")
    fig_pnl.add_vline(x=idx, line_width=1, line_dash="dot", line_color="black")
    fig_pnl.update_layout(autosize=True, margin=dict(l=50, r=20, t=30, b=20))

    fig_delta = _greek_fig("delta",  "Δ  (Delta)")
    fig_gamma = _greek_fig("gamma",  "Γ  (Gamma)")
    fig_vega  = _greek_fig("vega",   "Vega")
    fig_theta = _greek_fig("theta",  "Theta")
    fig_rho   = _greek_fig("rho",    "Rho")

    # ---------- 4) Tableau positions (TOTAL inclus) ----------
    snapshot_df = portfolio.state_at(idx)

    # Copie dédiée à l'AFFICHAGE (on tente une conversion numérique colonne par colonne)
    display_df = snapshot_df.copy()

    def _fmt_fr_3dec_num(x: float) -> str:
        # 1234567.89 -> "1 234 567,89"
        s = f"{x:,.3f}"
        return s.replace(",", " ").replace(".", ",")

    for c in display_df.columns:
        # Essaie de convertir la colonne en numérique
        as_num = pd.to_numeric(display_df[c], errors="coerce")
        # Indices où la conversion a réussi
        mask = as_num.notna()
        # Formate ces valeurs à 3 décimales (fr)
        display_df.loc[mask, c] = as_num[mask].map(_fmt_fr_3dec_num)
        # Les NaN deviennent chaîne vide pour l'affichage propre
        display_df[c] = display_df[c].where(display_df[c].notna(), "")

    table_data = display_df.to_dict("records")
    columns = [{"name": c, "id": c} for c in display_df.columns]


    # ---------- 5) Pages + responsive layout ----------
    if page_idx == 0:
        # Afficher le slider sur cette page
        slider_style = {"display": "block", "marginTop": "10px"}

        # Conteneur pleine hauteur (déduit l'espace du header/nav)
        content = html.Div(
            style={
                "display": "grid",
                "gridTemplateRows": "minmax(0, 50%) 1fr",  # 50% du panneau pour le graphe, puis le tableau
                "gap": "10px",
                "height": "calc(100vh - 140px)",  # ajuste si besoin selon ton header/padding
            },
            children=[
                dcc.Graph(
                    figure=fig_paths,
                    style={"height": "100%", "width": "100%"},
                    config={"responsive": True},
                ),
                dash_table.DataTable(
                    data=table_data,
                    columns=columns,
                    page_size=15,
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "auto",
                        "height": "100%",
                    },
                    style_cell={"textAlign": "right"},
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{option_type} = 'TOTAL'"},
                            "fontWeight": "bold",
                            "backgroundColor": "#f5f5f5",
                        }
                    ],
                ),
            ],
        )
    else:
        # Masquer le slider sur cette page
        slider_style = {"display": "none"}

        # Grille 2x3 pleine hauteur, chaque graphe prend 100%
        content = html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gridTemplateRows": "repeat(3, minmax(0, 1fr))",
                "gap": "10px",
                "height": "calc(100vh - 120px)", 
            },
            children=[
                dcc.Graph(figure=fig_pnl,   style={"height": "100%", "width": "100%"}, config={"responsive": True}),
                dcc.Graph(figure=fig_delta, style={"height": "100%", "width": "100%"}, config={"responsive": True}),
                dcc.Graph(figure=fig_gamma, style={"height": "100%", "width": "100%"}, config={"responsive": True}),
                dcc.Graph(figure=fig_vega,  style={"height": "100%", "width": "100%"}, config={"responsive": True}),
                dcc.Graph(figure=fig_theta, style={"height": "100%", "width": "100%"}, config={"responsive": True}),
                dcc.Graph(figure=fig_rho,   style={"height": "100%", "width": "100%"}, config={"responsive": True}),
            ],
        )

    return content, table_data, columns, slider_style



# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
