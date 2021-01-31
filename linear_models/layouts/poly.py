import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from util import hat, theta, subsup


metrics = html.Div(
    [
        html.H4("Metrics", style={"margin-bottom": "1.5em"}),
        dbc.Table(
            [
                html.Tr(
                    [
                        html.Th("MSE"),
                        html.Th("RMSE"),
                        html.Th("MAE"),
                        html.Th("$R^2$"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(id="mse_poly"),
                        html.Td(id="rmse_poly"),
                        html.Td(id="mae_poly"),
                        html.Td(id="r2_poly"),
                    ]
                ),
            ],
            style={"table-layout": "fixed"},
        ),
    ],
    className="col-lg-4 col-md-4 col-sm-12 pb-3",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "text-align": "center",
    },
)

parameters = html.Div(
    [
        html.H4("Parameters", style={"margin-bottom": "1.5em"}),
        dbc.Table(
            [
                html.Tr(
                    [
                        html.Td(
                            id="theta0_slider_value_poly",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta0_slider_poly",
                                min=-10,
                                max=10,
                                step=0.05,
                                value=0,
                                updatemode="drag",
                                className="custom-range",
                            )
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            id="theta1_slider_value_poly",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta1_slider_poly",
                                min=-10,
                                max=10,
                                step=0.05,
                                value=0,
                                updatemode="drag",
                                className="custom-range",
                            )
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            id="theta2_slider_value_poly",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta2_slider_poly",
                                min=-10,
                                max=10,
                                step=0.05,
                                value=0,
                                updatemode="drag",
                            )
                        ),
                    ]
                ),
            ],
            className="no-padding",
            borderless=True,
        ),
        html.Div(id="values_poly", style={"display": "none"}),
    ],
    className="col-lg-4 col-md-4 col-sm-12 pb-3",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "text-align": "center",
    },
)

prediction = html.Div(
    [
        html.H4("Prediction", style={"margin-bottom": "1.5em"}),
        html.Div(id="equation_poly", style={"padding-bottom": "25px"}),
        dbc.Table(
            [
                html.Tr(
                    [
                        html.Td(
                            id="prediction_slider_value_poly",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="prediction_slider_poly",
                                min=-3,
                                max=3,
                                step=0.05,
                                value=0,
                                updatemode="drag",
                                className="custom-range",
                            )
                        ),
                    ]
                ),
            ],
            className="no-padding",
            borderless=True,
        ),
    ],
    className="col-lg-4 col-md-4 col-sm-12 pb-3",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "text-align": "center",
    },
)

graph = html.Div(
    [
        dcc.Graph(
            id="graph_poly",
            figure=go.Figure(
                layout=dict(
                    margin=dict(t=0, r=0, l=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
            ),
            style={"width": "100%"},
            config={"staticPlot": True},
        ),
    ],
    className="col-lg-10 col-md-10 col-sm-12 pb-3",
)

graph_buttons = html.Div(
    [
        dbc.ButtonGroup(
            [
                dbc.Checklist(
                    options=[
                        {"label": "Residuals", "value": 1},
                    ],
                    value=[],
                    id="residual_switch_poly",
                    switch=True,
                    inline=True,
                    style={"padding-bottom": "10px"},
                ),
                dbc.Button(
                    "Best fit",
                    id="best_fit_button_poly",
                    className="btn btn-outline-primary",
                ),
                dbc.Button(
                    "New Dataset",
                    id="new_dataset_button_poly",
                    className="btn btn-outline-primary",
                ),
            ],
            vertical=True,
        ),
    ],
    className="col-lg-2 col-md-2 col-sm-12 pb-3",
    style={
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
    },
)

normal_equation = html.Div(
    id="normal_equation_poly",
    className="col-lg-6 col-md-8 col-sm-12 mb-4",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "overflow-x": "scroll",
    },
)

code_snippet = html.Div(
    id="code_example_poly",
    className="col-lg-6 col-md-8 col-sm-12",
    style={"margin-left": "auto", "margin-right": "auto"},
)

layout = html.Div(
    [
        html.Div(
            [
                html.H2("Polynomial regression", className="display-5"),
                html.P(
                    [
                        """The above is all nice and great, but there's one big problem that you've probably already noticed. Every time you click on "NEW DATASET" in the dashboard above, scikit-learn's """,
                        html.A(
                            html.Code("make_regression"),
                            target="blank_",
                            href="https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression",
                        ),
                        """ function creates a new dataset with a clear linear relationship between the dependent and explanatory variables. I do this on purpose because linear regression is a high bias model, which means that certain conditions must apply to the dataset. Unfortunately, not all relationships in life are linear, which is why you cannot simpy throw a linear regression on everything. So does this mean that linear regression is only applicable if the underlying dataset also has a linear structure 😢? No, it does not! At this point I'd like to introduce polynomial features, as they allow us to model a non-linear relationship in a linear way.""",
                    ]
                ),
                html.P(
                    f"""
    A polynomial feature is nothing more than an additional feature created from an existing feature. Let's say we have a dataset that has the shape of a parabola. It makes no sense to model the dataset using a simple linear regression. This is because the relationship is not linear but quadratic. To improve the fit of our line we extend the model by a polynomial feature which is just the squared value of our already existing feature. An additional feature means there is a further model parameter $\\theta_2$ which we'll have to optimize. As before we need nothing more than the normal equation to determine the best possible parameters $\\theta_0$, $\\theta_1$ and $\\theta_2$. This is because $ {hat(theta())} = {subsup(f"({subsup('X',sup='T')}X)",sup=-1)}{subsup('X',sup='T')}y $ works with one, ten or hundred features — it makes no difference. However, if your number of features is very large (e.g., more than 100,000), you should consider using gradient descent. This is because the normal equation inverts $ X^T X $, which can be quite computationally expensive with a large number of features.
    """
                ),
                html.P(
                    """
    In the following dashboard you can explore how a polynomial regression behaves. $\\theta_0$ and $\\theta_1$ have the same behavior as above: they change the vertical positioning and slope of the line. But what does $\\theta_2$ do? Find out by adjusting the sliders and creating new datasets. Also observe how the metrics change when you display the best fit.
    """
                ),
            ],
            className="bs-component",
        ),
        html.Div(
            [
                dbc.Row(
                    [
                        metrics,
                        parameters,
                        prediction,
                    ]
                ),
                dbc.Row(
                    [
                        graph,
                        graph_buttons,
                    ]
                ),
            ],
            className="bs-component",
        ),
        html.P(
            f"""
First things first: we can see that the scatter plot has the shape of a parabola. This means that unlike the first dashboard, our dataset has a quadratic rather than a linear relationship. $\\theta_2$ pushes and pulls the line inwards and outwards, respectively. For $ \\theta_2 = 0 $ we simply get a line. If we increase the absolute value of $\\theta_2$ the line more and more turns into the shape of a parabola. Feel free to play around with the sliders, display the residuals and try to reduce them. The prediction slider visualizes ${hat('y')}$ results for different $x_1$ values.
"""
        ),
        html.P(
            """
Polynomial features show how versatile linear regression models are, since small changes are already enough to create a whole new predictor. The calculation of the optimal model parameters is very similar to the univariate version and differs only by the additional model parameter $\\theta_2$ and the newly added polynomial feature $x_2$.
"""
        ),
        normal_equation,
        html.P(
            [
                "Once again, let's save us some work and use a combination of Numpy and scikit-learn's ",
                html.A(
                    html.Code("PolynomialFeatures"),
                    target="blank_",
                    href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html",
                ),
                " transformer instead:",
            ]
        ),
        code_snippet,
        html.P(
            [
                """
That's all I have to say about polynomial regressions. To be honest, it sounds a lot fancier than it actually is. Basically, you're just extending some existing features to make the model more flexible. Still, I find it fascinating that linear models can be used to explain quadratic or cubic relationships.
"""
            ]
        ),
        html.P(
            [
                """
At this point, a somewhat naive but valid question would be: why don't we just keep adding polynomial features to our model until we have a perfect model?  Couldn't we then explain every dataset perfectly? To some extent, this statement is true. By continuously adding polynomial features we are able to explain larger parts of the variance. This means that our prediction model gets better. Up until we are able to perfectly reproduce the dataset. Unfortunately, this approach doesn't help us much and is actually one of the best known pitfalls in machine learning. 
"""
            ]
        ),
    ]
)
