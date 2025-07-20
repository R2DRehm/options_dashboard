import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from portfolio.portfolio import OptionPosition, PortfolioSimulator
from utils.helpers import summarize_pnl, compute_var

app = Dash(__name__)
server = app.server

app.layout = html.Div(style={'display':'flex'}, children=[
    # Panel de contrôle
    html.Div(style={'width':'25%','padding':'20px','backgroundColor':'#f9f9f9','borderRight':'1px solid #ddd'}, children=[
        html.H2("Paramètres de simulation"),
        html.Label("Prix initial S0"), dcc.Input(id='s0', type='number', value=100.0, step=0.1),
        html.Label("Drift (μ)"), dcc.Input(id='mu', type='number', value=0.05, step=0.01),
        html.Label("Vol simulée (σ)"), dcc.Input(id='sigma_sim', type='number', value=0.2, step=0.01),
        html.Hr(),
        html.H2("Taux Vasicek"),
        html.Label("r0"), dcc.Input(id='r0', type='number', value=0.01, step=0.001),
        html.Label("κ_r"), dcc.Input(id='kappa_r', type='number', value=0.5, step=0.1),
        html.Label("θ_r"), dcc.Input(id='theta_r', type='number', value=0.05, step=0.01),
        html.Label("σ_r"), dcc.Input(id='sigma_r', type='number', value=0.01, step=0.001),
        html.Hr(),
        html.H2("Volatilité OU"),
        html.Label("σ0"), dcc.Input(id='x0', type='number', value=0.2, step=0.01),
        html.Label("κ_v"), dcc.Input(id='kappa_v', type='number', value=1.0, step=0.1),
        html.Label("θ_v"), dcc.Input(id='theta_v', type='number', value=0.2, step=0.01),
        html.Label("ξ"), dcc.Input(id='xi', type='number', value=0.1, step=0.01),
        html.Hr(),
        html.H2("Général"),
        html.Label("Horizon T (années)"), dcc.Input(id='T', type='number', value=1.0, step=0.1),
        html.Label("Pas de temps N"), dcc.Input(id='N', type='number', value=252, step=1),
        html.Label("Scénarios M"), dcc.Input(id='M', type='number', value=10, step=1),
        html.Label("Graine"), dcc.Input(id='seed', type='number', value=42, step=1),
        html.Hr(),
        html.H2("Option"),
        html.Label("Type"),
        dcc.Dropdown(id='option_type', options=[{'label':'Call','value':'call'}, {'label':'Put','value':'put'}], value='call'),
        html.Label("Strike K"), dcc.Input(id='K', type='number', value=100, step=1),
        html.Label("Quantité"), dcc.Input(id='quantity', type='number', value=1, step=1),
        html.Br(), html.Button("Simuler", id='simulate', n_clicks=0)
    ]),

    # Panel de sortie
    html.Div(style={'width':'75%','padding':'20px'}, children=[
        html.Div(id='output-content')
    ])
])

@app.callback(
    Output('output-content','children'),
    Input('simulate','n_clicks'),
    State('s0','value'), State('mu','value'), State('sigma_sim','value'),
    State('r0','value'), State('kappa_r','value'), State('theta_r','value'), State('sigma_r','value'),
    State('x0','value'), State('kappa_v','value'), State('theta_v','value'), State('xi','value'),
    State('T','value'), State('N','value'), State('M','value'), State('seed','value'),
    State('option_type','value'), State('K','value'), State('quantity','value')
)
def update_output(n_clicks, S0, mu, sigma_sim, r0, kappa_r, theta_r, sigma_r,
                  x0, kappa_v, theta_v, xi, T, N, M, seed,
                  option_type, K, quantity):
    if n_clicks == 0:
        return html.Div("Appuyez sur 'Simuler' pour démarrer.")

    # Simulation
    pos = OptionPosition(option_type, K, T, quantity)
    sim = PortfolioSimulator(
        positions=[pos],
        S_params={'S0': S0, 'mu': mu, 'sigma': sigma_sim},
        vol_params={'x0': x0, 'kappa': kappa_v, 'theta': theta_v, 'xi': xi},
        rate_params={'r0': r0, 'kappa': kappa_r, 'theta': theta_r, 'sigma': sigma_r},
        T=T, N=N, M=M, seed=int(seed)
    )
    df = sim.run()

    # Table des données brutes
    data_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

    # Choisir premier scénario pour graphiques
    sce = df['scenario'].unique()[0]
    df_sce = df[df['scenario'] == sce]

    price_fig = px.line(df_sce, x='t', y='S', title='Trajectoire de prix')
    pnl_fig = px.line(df_sce, x='t', y='pnl', title='PnL le long du temps')
    greek_figs = [px.line(df_sce, x='t', y=g, title=f'Évolution de {g}') for g in ['delta', 'gamma', 'vega', 'theta', 'rho']]

    # Résumé des PnL
    summary_df = summarize_pnl(df)
    summary_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in summary_df.reset_index().columns],
        data=summary_df.reset_index().to_dict('records')
    )
    var_value = compute_var(df)

    # Assemblage du layout de sortie
    return html.Div(children=[
        html.H3("Données brutes"), data_table,
        html.H3("Trajectoire de prix"), dcc.Graph(figure=price_fig),
        html.H3("PnL"), dcc.Graph(figure=pnl_fig),
        html.H3("Grecques"), *[dcc.Graph(figure=fig) for fig in greek_figs],
        html.H3("Résumé des PnL finaux"), summary_table,
        html.P(f"VaR (5%): {var_value:.2f}")
    ])

if __name__ == '__main__':
    app.run(debug=True)
