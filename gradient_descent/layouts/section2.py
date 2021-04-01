import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

section2_layout = html.Div([
    dbc.Col([
        html.H2('A less simple example', id='a_less_simple_example'),
        html.P("""
        Most real examples deal with more than one parameter, so we will abandon this assumption. In the following example, we have the same underlying data set, but we will try to find the optimal solution by finding the line with the best slope and intercept. The following interactive diagram should once again give you an idea of the type of problem we are dealing with and which best fitting line to look for.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        dbc.Row(
            [
                dbc.Col([
                    html.Div([
                        html.Div(id='dashboard3_div_mse', style={'padding-bottom':'2rem', 'text-align':'center'}),
                        html.Div([
                            html.P(id='dashboard3_p_theta0', style={'margin-bottom':'0px'}),
                            dcc.Slider(id='dashboard3_slider_theta0', min=0, max=10, step=0.1, value=1),
                            html.P(id='dashboard3_p_theta1', style={'margin-bottom':'0px'}),
                            dcc.Slider(id='dashboard3_slider_theta1', min=-10, max=10, step=0.1, value=1),
                        ], style={'text-align':'center'}),
                        html.Div([
                                dbc.Button("Best fit", color='secondary', id='dashboard3_button_bestfit'),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Residuals", "value": 1},
                                    ],
                                    value=[],
                                    id="dashboard3_checklist_residuals",
                                    switch=True,
                                    style={'padding-left':2}
                                ),
                        ], style={'display':'flex', 'align-items':'center'}),
                    ]),
                ], width=12, lg=2, sm=12, className='d-flex align-items-center justify-content-center container'),
                dbc.Col([
                    html.Div(dcc.Graph(id='dashboard3_graph_regression')),
                ], width=12, lg=10, sm=12)
            ]
        ),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component center'), 
    dbc.Col([
        html.P("""
        I am aware that most real-life cases also deal with more than two parameters, but for Gradient Descent it makes no difference whether you have two or a hundred model parameters. This is because the underlying concept remains the same and only the computational complexity increases. Let's find out what Gradient Descent looks like with two parameters and what changes from the previous example.
        """),
        html.P("""
        We still have our MSE cost function: $$ \\textrm{MSE}(\\theta) = \\frac{{1}}{{2m}}\\sum_{{i=1}}^{{m}}{{(\\theta^Tx^i-y^i)^2}}, \\: \\textrm{with} \\: \\theta=\\begin{pmatrix}\\theta_0\\\\\\theta_1\\end{pmatrix} $$ The only difference is that we loosen our assumption about a fixed $\\theta_0$ value and treat it as a real model parameter. Pretty much the same way we treated $\\theta_1$ in the previous example. Here is the first difference: since we need to calculate the partial derivative of the cost function with respect to each model parameter, we have an additional equation: $$ \\frac{{\\partial}}{{\\partial{{\\theta_0}}}}\\textrm{MSE}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{0}}^{{i}} $$ $$ \\frac{{\\partial}}{{\\partial{{\\theta_1}}}}\\textrm{MSE}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{1}}^{{i}} $$ Unlike the first example, we are now dealing with an actual gradient and cannot further oversimplify $\\frac{{\\partial}}{{\\partial{{\\theta_0}}}}\\textrm{MSE}(\\theta), \\frac{{\\partial}}{{\\partial{{\\theta_1}}}}\\textrm{MSE}(\\theta)$ to the slope of the cost function. However, everything else remains the same. We take a random guess for the initialization of $\\theta_0$ and $\\theta_1$, measure the local gradient of the cost function with respect to each model parameter, and update the parameters based on the learning rate. In this way, we slowly move towards the descending gradient. Interested to see what this example looks like? Check out the interactive chart below. Learning rate, best fit and reset are the same as in the previous example, but you will quickly notice the differences. In fact, everything looks a bit fancier with an additional parameter, but it will feel very natural after some time.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(id='dashboard4_div_mse', className='pb-3'),
                    html.Div(id='dashboard4_div_thetas', className='pb-4'),
                    dbc.ButtonGroup([
                        html.Div([
                            html.P('Learning rate $\\eta$ ($[0,2]$)', style={'margin-bottom':'0px'}),
                            dbc.Input(id="dashboard4_input_eta", type="number", value=0.1, step=0.1, min=0, max=2, style={'text-align':'center'}),
                        ], style={'text-align':'center'}),
                        dbc.Button(id='dashboard4_button_nextstep', color='secondary', children='Next step'),
                        dbc.Button(id='dashboard4_button_reset', color='secondary', children='Reset'),
                        # dbc.Input(id="dashboard4_input_eta", type="number", placeholder="Learning rate", value=0.1, step=0.1),
                    ], vertical=True),
                    html.Div(
                        dbc.Checklist(
                            options=[
                                {"label": "Scale X", "value": 1},
                            ],
                            value=[],
                            id="dashboard4_checklist_scalex",
                            switch=True,
                        ), style={'width':'100%'},# className='pb-2',
                    ),
                ], style={'text-align':'center'}),
            ], xs=12, sm=12, md=12, lg=2, className='d-flex align-items-center justify-content-center container'),
            dbc.Col([
                html.Div(dcc.Graph(id='dashboard4_graph_regression')),
            ], xs=12, sm=12, md=12, lg=5),
            dbc.Col([
                html.Div(dcc.Graph(id='dashboard4_graph_lossfunction')),
            ], xs=12, sm=12, md=12, lg=5),
        ]),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component center', id='interactive2'), 
    dbc.Col([
        html.P([
            "When you click ",
            html.Code('NEXT STEP'),
            ", you should have the same feeling as the previous example: Step by step you are approaching the optimal solution. The only difference in the diagram on the left is that the Gradient Descent updates both the intercept and the slope with each successive iteration. But wait 😱, what the hell happened to our bowl-shaped looking cost function chart on the right? Suddenly we're seeing a fancy 3d chart with all sorts of planes and contours that make things look a lot more complicated."
        ]),
        html.P("""
        Fortunately, this is not the case. In the previous example, we took care of one parameter while paying attention to the cost function. Now we take care of two parameters while paying attention to the cost function. That's a total of three axes, which is why we're working in 3d rather than 2d space. Also, with two parameters, our cost function becomes a surface, since the MSE depends on both $\\theta_0$ and $\\theta_1$. The same thing happens with our gradient. In the previous example, our gradient was a simple line defined only by the slope of the cost function. Now the gradient is a tangent plane that touches the surface of the cost function at $(\\theta_0, \\theta_1)$.
        """),
        html.P("""
        With regard to the learning rate, all of the four previously mentioned scenarios also apply in this case, which means that the learning rate should be neither too large nor too small. In the following two scenarios, we will see that besides the learning rate, the scaling of the input features also plays a crucial role.
        """),
        html.Ul([
            html.Li([
                html.A('Unscaled input features ($\\theta_0=15,\\theta_1=-15,\\eta=0.5,\\textrm{scalex}=\\textrm{False}$):', href="#interactive2", id="dashboard4_href_li1", style={"color": "blue", "font-weight": "bold",}),
                html.P('Depending on the random initialization, Gradient Descent takes a complex path with many iterations.')
            ]),
            html.Li([
                html.A('Scaled input features ($\\theta_0=15,\\theta_1=-15,\\eta=0.5,\\textrm{scalex}=\\textrm{True}$):', href="#interactive2", id="dashboard4_href_li2", style={"color": "blue", "font-weight": "bold",}),
                html.P('Independent of the random initialization, Gradient Descent quickly finds its way to the global optimum.')
            ]),
        ]),
        html.P([
            "The problem can be clearly visualized in the chart above by playing around with the ",
            html.Code('Scale X'),
            " slider. To understand why feature scaling is so important, let's focus on the contour chart located at the bottom of the right chart. The contour chart contains the same information as the cost function surface, only in 2d. If you move your mouse pointer over the surface, the contour plot below it will show you the contour of the cost function transferred into 2d space. The most important thing here is the shape of the contour. In the unscaled case, the contour is an elongated bowl, while in the scaled example, it is a neat circle. The implication for Gradient Descent is that without prior scaling, the path to the optimal solution is often complicated and cornered. After prior scaling, on the other hand, Gradient Descent follows a straight line to the best possible solution and converges much faster."
        ]),
        html.P("""
        Once again we look at the underlying math. Try to pay special attention to the two partial derivatives $\\frac{{\\partial}}{{\\partial{{\\theta_0}}}}\\textrm{MSE}(\\theta), \\frac{{\\partial}}{{\\partial{{\\theta_1}}}}\\textrm{MSE}(\\theta)$ and how in this example both parameters $\\theta_0, \\theta_1$ are adjusted step by step.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        html.Div([
            dbc.Col([
                dbc.Table(id='dashboard4_table_math', bordered=False),
            ], xs=12, sm=12, md=12, lg=6, style={'margin-left':'auto', 'margin-right':'auto', 'overflow-x':'scroll'}),
        ]),
        html.Div(dbc.Button(id='dashboard4_button_nextstep_table', color='secondary', children='Next step'), className='pt-1', style={'text-align':'center'}),
        html.Div(id='dashboard4_div_thetainit', style={'display': 'none'}),
        html.Div(id='dashboard4_div_thetahist', style={'display': 'none'}),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component center'),
    dbc.Col([
        html.P("""
        Congratulations! If you succeeded up to here, you have mastered Gradient Descent. To be precise, we have dealt with the most general variant, namely Batch Gradient Descent. In the previous examples we could see that Batch Gradient Descent is an excellent tool to solve optimization problems, but the algorithm also has some weaknesses. In the next section we will deal with the pitfalls of Batch Gradient Descent and some possible alternatives.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
])





