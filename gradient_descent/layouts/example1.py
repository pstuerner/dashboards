import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

layout = html.Div([
    dbc.Row(
        [
            dbc.Col([
                html.Div([
                    html.Div(id='example1_div_mse', style={'padding-bottom':'2rem', 'text-align':'center'}),
                    html.Div([
                        html.P(id='example1_p_theta1', style={'margin-bottom':'0px'}),
                        dcc.Slider(id='example1_slider_theta1', min=-10, max=10, step=0.1, value=1),
                    ], style={'text-align':'center'}),
                    html.Div([
                            dbc.Button("Best fit", color='secondary', id='example1_button_bestfit'),
                            dbc.Checklist(
                                options=[
                                    {"label": "Residuals", "value": 1},
                                ],
                                value=[],
                                id="example1_checklist_residuals",
                                switch=True,
                                style={'padding-left':2}
                            ),
                    ], style={'display':'flex', 'align-items':'center'}),
                ]),
            ], width=12, lg=2, sm=12, className='pb-4 d-flex align-items-center justify-content-center container'),
            dbc.Col([
                html.Div(dcc.Graph(id='example1_graph_regression')),
            ], width=12, lg=10, sm=12, className='pb-4')
        ], className='pt-4 pb-4'
    ),
])