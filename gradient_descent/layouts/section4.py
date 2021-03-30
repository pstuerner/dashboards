import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

lr_options = [
        {'label':'Constant', 'value':'constant'},
        {'label':'Time-based', 'value':'time'},
        {'label':'Stepwise', 'value':'step'},
    ]

bgd_lr_dropdown = dcc.Dropdown(
    id='bgd_lr_dropdown',
    options=lr_options,
    value='constant',
    clearable=False,
    searchable=False,
),

sgd_lr_dropdown = dcc.Dropdown(
    id='sgd_lr_dropdown',
    options=lr_options,
    value='time',
    clearable=False,
    searchable=False,
),

mbgd_lr_dropdown = dcc.Dropdown(
    id='mbgd_lr_dropdown',
    options=lr_options,
    value='time',
    clearable=False,
    searchable=False,
),

section4_layout = html.Div([
    dbc.Col([
        html.H2('Gradient Descent Racetrack', id='gradient_descent_racetrack'),
        html.P("""
        It's time to compare the three Gradient Descent variations we' ve just discussed and get a sense of how they perform in different scenarios. So, let's hit the racetrack! But before you place a bet on any of the three horses let me first say a few words about the following dashboard.
        """),
        html.P("""
        As in the previous examples, this is a linear regression problem with one dependent variable and two explanatory variables. The underlying data as well as the current regression model for each gradient descent step are shown in the left graph. Pay attention: since we are comparing three different algorithms, we get three different results for each step. In the right chart we see the cost function as a contour plot for different theta0, theta1 values. The red diamond is the starting point (random initialization) and all three gradient descent variants try step by step to reach the green X, which is the global optimum. The dropdown menu below '# instances' controls the size of the data set. To save you from having to perform many individual gradient descent steps, you can increase the counter by clicking on 'Next step(s)' and thus perform multiple iterations with a single click. In the table you can set the respective learning schedules and the batch size for Mini-Batch Gradient Descent. To ensure consistent scenarios, the learning schedules, batch size and data set size cannot be changed once the first gradient descent step has been calculated. Just click the reset button to start a new experiment and try out new options. Just give it a shot. After the dashboard, I discuss a few noteworthy scenarios that highlight differences and similarities.
        """),
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(id='racetrack_div_mse', className='pb-3'),
                    dbc.ButtonGroup([
                        html.P('# instances', className='center', style={'text-align':'center'}),
                        dcc.Dropdown(
                            id='racetrack_dropdown_size',
                            options=[
                                {'label':100, 'value':100},
                                {'label':500, 'value':500},
                                {'label':1000, 'value':1000},
                                {'label':5000, 'value':5000},
                                {'label':10000, 'value':10000},
                            ],
                            value=5000,
                            clearable=False,
                            searchable=False,
                            style={'width':'100%', 'margin-bottom':'1em'}
                        ),
                        html.Br(),
                        dbc.Input(id="racetrack_input_steps", type="number", placeholder="# Steps", value=5, step=1, min=1, max=50, style={'text-align':'center'}),
                        dbc.Button(id='racetrack_button_nextstep', color='secondary', children='Next step(s)'),
                        dbc.Button(id='racetrack_button_reset', color='secondary', children='Reset'),
                    ], vertical=True),
                ], style={'text-align':'center'}),
            ], xs=12, sm=12, md=12, lg=2, className='d-flex align-items-center justify-content-center container'),
            dbc.Col([
                html.Div(dcc.Graph(id='racetrack_graph_regression')),
            ], xs=12, sm=12, md=12, lg=5),
            dbc.Col([
                html.Div(dcc.Graph(id='racetrack_graph_contour')),
            ], xs=12, sm=12, md=12, lg=5),
        ]),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component center', id='interactive4'), 
    dbc.Col([
        dbc.Row([
            dbc.Col([
                dbc.Table([
                    html.Tr([
                        html.Th(),
                        html.Th(html.B('Learning schedule')),
                        html.Th(html.B('Batch size')),
                        html.Th(html.B('Epoch')),
                        html.Th(html.B('Batch')),
                        html.Th(html.B('Learning rate')),
                        html.Th(html.B('MSE')),
                        html.Th(html.B('Rank')),
                        html.Th(html.B('Avg time')),
                        html.Th(html.B('Rank')),
                    ]),
                    html.Tr([
                        html.Td(html.B('BGD'), style={'text-align':'center'}),
                        html.Td(bgd_lr_dropdown),
                        html.Td(id='bgd_batchsize'),
                        html.Td(id='bgd_epoch'),
                        html.Td(id='bgd_batch'),
                        html.Td(id='bgd_learningrate'),
                        html.Td(id='bgd_mse'),
                        html.Td(id='bgd_mse_rank'),
                        html.Td(id='bgd_time'),
                        html.Td(id='bgd_time_rank'),
                    ], id='bgd_tr'),
                    html.Tr([
                        html.Td(html.B('SGD'), style={'text-align':'center'}),
                        html.Td(sgd_lr_dropdown),
                        html.Td(id='sgd_batchsize'),
                        html.Td(id='sgd_epoch'),
                        html.Td(id='sgd_batch'),
                        html.Td(id='sgd_learningrate'),
                        html.Td(id='sgd_mse'),
                        html.Td(id='sgd_mse_rank'),
                        html.Td(id='sgd_time'),
                        html.Td(id='sgd_time_rank'),
                    ]),
                    html.Tr([
                        html.Td(html.B('MBGD'), style={'text-align':'center'}),
                        html.Td(mbgd_lr_dropdown),
                        html.Td(id='mbgd_batchsize_td'),
                        html.Td(id='mbgd_epoch'),
                        html.Td(id='mbgd_batch'),
                        html.Td(id='mbgd_learningrate'),
                        html.Td(id='mbgd_mse'),
                        html.Td(id='mbgd_mse_rank'),
                        html.Td(id='mbgd_time'),
                        html.Td(id='mbgd_time_rank'),
                    ]),
                ], bordered=False),
            ], xs=12, sm=12, md=12, lg=12),
        ]),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component center'), 
])