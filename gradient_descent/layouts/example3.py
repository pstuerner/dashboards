import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(id='example3_div_mse', className='pb-2'),
                html.Div(id='example3_div_thetas', className='pb-2'),
                dbc.ButtonGroup([
                    dbc.Button(id='example3_button_nextstep', color='secondary', children='Next step'),
                    dbc.Button(id='example3_button_reset', color='secondary', children='Reset'),
                    dbc.Input(id="example3_input_eta", type="number", placeholder="Learning rate $\eta$", value=0.1, step=0.1),
                ], vertical=True),
            ], style={'text-align':'center'}),
        ], xs=12, sm=12, md=12, lg=2, className='pb-4 d-flex align-items-center justify-content-center container'),
        dbc.Col([
            html.Div(dcc.Graph(id='example3_graph_regression')),
        ], xs=12, sm=12, md=12, lg=5),
        dbc.Col([
            html.Div(dcc.Graph(id='example3_graph_lossfunction')),
        ], xs=12, sm=12, md=12, lg=5),
    ]),
    html.Div(id='example3_div_theta1init', style={'display': 'none'}),
    html.Div(id='example3_div_theta1hist', style={'display': 'none'}),
])