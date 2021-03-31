import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from util import theta0_best

section1_layout = html.Div([
    dbc.Col([
        html.H2('(Batch) Gradient Descent', id='batch_gradient_descent'),
        html.P("""
        Some machine learning methods sound catchy, but you have absolutely no idea what they do after you first hear about them. I mean, why should I care about a Random Forest and what am I supposed to do with a Ridge and Lasso? On the other hand, there are methods with really precise names, of which you kind of already have an idea after you read them for the first time. In my opinion, gradient descent belongs to the latter.
        """),
        html.P("""
        That's because gradient descent is about descending something called the gradient. Nothing more, nothing less. Things get even simpler once you realize that "gradient" is just a fancy term for the derivative or rate of change of a function. As is often the case in machine learning, the function of interest is a cost function that measures the overall fit of your model. So the main idea behind gradient descent is to iteratively change and revise model parameters to minimize a cost function. This is done by calculating the local gradient of the cost function with respect to each model parameter and going in the direction of the descending gradient. Once the gradient is zero (or really close to zero), you have reached an optimum! Let's find out what all this looks like by examining a really simple regression problem.
        """),
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'),  
    dbc.Col([
        html.H2('A really simple example', id='a_really_simple_example'),
        html.P("""
        Let’s work on a dead simple regression problem including one explanatory variable $X$ and one dependent variable $y$. Our goal is to find a linear model that describes the underlying data best and can be used for future predictions. If you’re not familiar with linear models yet make sure to read through one of my other dashboards that covers all kinds of linear models in greater detail. The following interactive chart should give you an idea about what kind of problem we’re dealing with and which assumptions I took to simplify this example as much as possible.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'),
    dbc.Col([    
        dbc.Row(
            [
                dbc.Col([
                    html.Div([
                        html.Div(id='dashboard1_div_mse', style={'padding-bottom':'2rem', 'text-align':'center'}),
                        html.Div([
                            html.P(id='dashboard1_p_theta1', style={'margin-bottom':'0px'}),
                            dcc.Slider(id='dashboard1_slider_theta1', min=-10, max=10, step=0.1, value=1),
                        ], style={'text-align':'center'}),
                        html.Div([
                                dbc.Button("Best fit", color='secondary', id='dashboard1_button_bestfit'),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Residuals", "value": 1},
                                    ],
                                    value=[],
                                    id="dashboard1_checklist_residuals",
                                    switch=True,
                                    style={'padding-left':2}
                                ),
                        ], style={'display':'flex', 'align-items':'center'}),
                    ]),
                ], width=12, lg=2, sm=12, className='d-flex align-items-center justify-content-center container'),
                dbc.Col([
                    html.Div(dcc.Graph(id='dashboard1_graph_regression')),
                ], width=12, lg=10, sm=12)
            ],
        ),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component'),
    dbc.Col([
        html.P("""
        As you can see, our problem is limited to finding the best possible $\\theta_1$ parameter. This is because I have already calculated the best possible $\\theta_0$ parameter and set it for the entire example. The reason for this is that it allows me to plot the gradient and the cost function in a 2D scatter plot, which really helps to understand how gradient descent works step by step. So don't worry about $\\theta_0$ now, we will talk about how to use gradient descent to optimize multiple parameters later.
        """),
        html.P("""
        If you click on the BEST FIT button, you will see what our final result should be. There are several ways to figure out which value for $\\theta_1$ gives the best fitting line (see, for example, my dashboard for linear models for the normal equation), and gradient descent is one of them. As we already know from above, there are two special ingredients we need before we can get started: the cost function and its gradient.
        """),
        html.P(f"""
        There are a variety of cost functions, but we will stick to the mean squared error (MSE) since it is simple, convex (we will cover this later), and differentiable. This is what MSE looks like: $$ \\textrm{{MSE}}(\\theta) = \\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}{{(\\theta^Tx^i-y^i)^2}}, \\: \\textrm{{with}} \\: \\theta=\\begin{{pmatrix}}\\theta_0^{{\\textrm{{best}}}}\\\\\\theta_1\\end{{pmatrix}}=\\begin{{pmatrix}}{round(theta0_best,2)}\\\\\\theta_1\\end{{pmatrix}} $$ However, there is a small modification that will make our lives much easier, and that is to multiply the entire equation by $\\frac{{1}}{{2}}$: $$ \\textrm{{MSE}}(\\theta) = \\frac{{1}}{{2m}}\\sum_{{i=1}}^{{m}}{{(\\theta^Tx^i-y^i)^2}} $$ The reason for this is that it makes the math, in our case the derivative of the cost function with respect to each model parameter, easier to handle. Fortunately, our simple example contains only one model parameter, which means that the gradient and the slope of the cost function are exactly the same. The derivative of MSE with respect to $\\theta_1$ looks like this: $$ \\textrm{{MSE}}_{{\\theta_1}}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{1}}^{{i}} $$ As you can see, the previously added $\\frac{{1}}{{2}}$ cancels out due to the exponent rule applied to $(\\theta^Tx^i-y^i)^2$. So adding $\\frac{{1}}{{2}}$ is just for convenience and makes the derivation look prettier. It doesn't matter for the result because the minimization is unaffected by constants.
        """),
        html.P("""
        We are almost done and there are only two more things to mention before you can explore gradient descent step by step. The first is to tell gradient descent where to start, i.e. which parameter we use first to evaluate the slope of the cost function. The solution is simple: just take a random guess. Start anywhere and gradient descent will figure out which direction to go. The last missing ingredient is the so-called learning rate $\\eta$. It is certainly one of the most important parameters, as it determines the size of the steps that gradient descent takes at each iteration. For the learning rate, neither too large nor too small is the correct solution. If the learning rate is too small, gradient descent will require many iterations and significantly more time to reach the optimal solution. If the learning rate is too large, then gradient descent may even overshoot and move away from the optimal solution, meaning that the algorithm will not converge. So the sweet spot is in the middle: small enough for guaranteed convergence, large enough for as few iterations as possible.
        """),
        html.P("""
        That's it! Now you have everything in your toolbox that you need to understand the following interactive charts. Each time you click the RESET button, a random initialization of the $\\theta_1$ parameter is performed in the left diagram. The right plot contains the MSE cost function for different values of $\\theta_1$ and the resulting gradient (aka slope). Once you click NEXT STEP, gradient descent calculates the slope for the current $\\theta_1$ value, combines it with the learning rate, and gives us the new parameter value of $\\theta_1$. Be sure to play around with different learning rates and see if you can spot a difference.
        """),
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(id='dashboard2_div_mse', className='pb-2'),
                    html.Div(id='dashboard2_div_thetas', className='pb-2'),
                    dbc.ButtonGroup([
                        dbc.Button(id='dashboard2_button_nextstep', color='secondary', children='Next step'),
                        dbc.Button(id='dashboard2_button_reset', color='secondary', children='Reset'),
                        dbc.Input(id="dashboard2_input_eta", type="number", placeholder="Learning rate $\eta$", value=0.1, step=0.1),
                    ], vertical=True),
                ], style={'text-align':'center'}),
            ], xs=12, sm=12, md=12, lg=2, className='d-flex align-items-center justify-content-center container'),
            dbc.Col([
                html.Div(dcc.Graph(id='dashboard2_graph_regression')),
            ], xs=12, sm=12, md=12, lg=5),
            dbc.Col([
                html.Div(dcc.Graph(id='dashboard2_graph_lossfunction')),
            ], xs=12, sm=12, md=12, lg=5),
        ]),
        html.Div(id='dashboard2_div_theta1init', style={'display': 'none'}),
        html.Div(id='dashboard2_div_theta1hist', style={'display': 'none'}),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component', id='interactive1'),
    dbc.Col([
        html.P("""
        The first thing we see is that gradient descent is an iterative process. We start somewhere far away and approach the optimal solution with each subsequent step. This is done by iteratively setting and updating $\\theta_1$ based on the value of the gradient and the learning rate. If you track the gradient in the graph on the right, you will see that it is actually approaching zero as the red line gets flatter with each step. To give you a feel for the learning rate, let's cover four of the most common scenarios. Be sure to click on the hyperlinks and run a few next steps on the interactive graph to see what it looks like.
        """),
        html.Ul([
            html.Li([
                html.A('High learning rate, slow convergence ($\\theta_1=4.9, \\eta=1.5$):', href="#interactive1", id="dashboard2_href_li1", style={"color": "blue", "font-weight": "bold",}),
                html.P('A higher learning rate does not mean that the algorithm converges faster. The algorithm may not converge at all, or it may take more iterations while bouncing around the optimal solution.')
            ]),
            html.Li([
                html.A('High learning rate, no convergence ($\\theta_1=1.8, \\eta=2$):', href="#interactive1", id="dashboard2_href_li2", style={"color": "blue", "font-weight": "bold",}),
                html.P('Sometimes gradient descent is just very motivated, shooting past the optimal $\\theta_1$ and deviating from the best solution with each successive step.')
            ]),
            html.Li([
                html.A('Low learning rate, slow convergence ($\\theta_1=-1, \\eta=0.05$):', href="#interactive1", id="dashboard2_href_li3", style={"color": "blue", "font-weight": "bold",}),
                html.P('Gradient descent is guaranteed to converge if your cost function is convex and the learning rate is small enough. That is, as you decrease the learning rate, the time to convergence also increases, since each update is just a baby step. Expect some waiting time if your random initialization is far from the optimal solution and the learning rate is too low.')
            ]),
            html.Li([
                html.A('Good learning rate, fast convergence ($\\theta_1=7.5, \\eta=0.4$):', href="#interactive1", id="dashboard2_href_li4", style={"color": "blue", "font-weight": "bold",}),
                html.P('As you can see, the sweet spot is somewhere in the middle. Just a few iterations are enough to get very close to the optimal solution. The left graph is really close to the best fit, while the gradient is close to zero, as we can see from the flat red line in the right graph.')
            ]),
        ]),
        html.P("""
        Great! This is what simple one-parameter gradient descent looks like from a visual point of view. Before we continue, let's take a quick look at the math behind the graph. The following section shows the equations for each gradient descent step. It is dynamic, updates with each step, and adjusts the calculations to match the corresponding learning rate. Feel free to change the learning rate, reset all gradient descent steps, and follow along each time you click NEXT STEP. Try to pay special attention to how the previous value of $\\theta_1$, the learning rate $\\eta$, and the gradient $\\text{MSE}_{{\\theta_1}}$ interact to result in a new $\\theta_1$.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'),
    dbc.Col([
        dbc.Col([
            dbc.Table(id='dashboard2_table_math', bordered=False),
        ], xs=12, sm=12, md=12, lg=6, style={'margin-left':'auto', 'margin-right':'auto', 'overflow-x':'scroll'}),
        html.Div(dbc.Button(id='dashboard2_button_nextstep_table', color='secondary', children='Next step'), className='pt-1', style={'text-align':'center'}),
    ], xs=12, sm=12, md=12, lg=12, className='bs-component'),
    dbc.Col([
        html.P("""
        That's it for the simple example. Not so complicated when you visualize each step and look at the underlying math for each iteration, right? Let's build on what we just learned and extend the previous example to make it a little more complex.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'),
])