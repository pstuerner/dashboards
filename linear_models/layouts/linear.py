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
                        html.Td(id="mse_linear"),
                        html.Td(id="rmse_linear"),
                        html.Td(id="mae_linear"),
                        html.Td(id="r2_linear"),
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
                            id="theta0_slider_value_linear",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta0_slider_linear",
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
                            id="theta1_slider_value_linear",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta1_slider_linear",
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
            ],
            className="no-padding",
            borderless=True,
        ),
        html.Div(id="values_linear", style={"display": "none"}),
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
        html.Div(
            id="equation_linear",
            style={"padding-bottom": "25px"},
        ),
        dbc.Table(
            [
                html.Tr(
                    [
                        html.Td(
                            id="prediction_slider_value_linear",
                            style={
                                "width": "1%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="prediction_slider_linear",
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
            id="graph_linear",
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
                    id="residual_switch_linear",
                    switch=True,
                    inline=True,
                    style={"padding-bottom": "10px"},
                ),
                dbc.Button(
                    "Best fit",
                    id="best_fit_button_linear",
                    className="btn btn-outline-primary",
                ),
                dbc.Button(
                    "New Dataset",
                    id="new_dataset_button_linear",
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
    id="normal_equation_linear",
    className="col-lg-6 col-md-8 col-sm-12 mb-4",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "overflow-x": "scroll",
    },
)

code_snippet = html.Div(
    id="code_example_linear",
    className="col-lg-6 col-md-8 col-sm-12",
    style={"margin-left": "auto", "margin-right": "auto"},
)

layout = html.Div(
    [
        html.Div(
            [
                html.H2("Univariate linear regression", className="display-5"),
                html.P(
                    """
    Univariate linear regression involves modeling or predicting the numerical value of one variable based on the data of another variable. What will be the gross domestic product in the next year if the production numbers remain the same? What is the weight of this person at a height of 1.80m? Do people who spend more time on social media generally feel more insecure? If we try to explain the result of a dependent variable (GDP, weight, feeling of insecurity) on the basis of an explanatory variable (production numbers, height, time spent on social media) we are dealing with a univariate linear regression.
    """
                ),
                html.P(
                    [
                        f"The heart of a (univariate) linear regression is the following equation: ",
                        f"$$ {hat('y')} = {theta(sup='T')} x $$",
                        f"""${hat('y')}$ is the predicted value of the dependent variable (e.g. GDP). ${theta()}$ is the model's parameter vector which, in the univariate case, consist of the bias term ${theta(sub=0)}$ and the weight of the explanatory variable ${theta(sub=1)}$ (e.g. production numbers). $x$ is the model's feature vector which includes all explanatory variables. Remember that to properly embed the bias term you have to make sure that the first column of the feature vector is made up of ones. Now the thing is, this is much easier to understand when you can graphically see what's going on and manipulate the parameters yourself. Feel free to play around with the ${theta(sub=0)}$ and ${theta(sub=1)}$ sliders in the dashboard below. Watch how the line moves and repeat the process with different datasets. If you want to know what the perfect line for the underlying dataset looks like click on the "BEST FIT" button. Ignore the residuals slider and metrics for now, we'll get to them in a moment.""",
                    ]
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
        html.Div(
            [
                html.P(
                    f"""
    As you can see ${theta(sub=0)}$ simply changes the vertical positioning while ${theta(sub=1)}$ changes the slope of the line. But how to find the line that best approximates the scatter plot? The answer is to find the line to which the distance to all points is the smallest. This can be achieved by using different cost functions that quantify the fit of a line. Among the most common cost functions are:
    """
                ),
                html.Ul(
                    [
                        html.Li(
                            "Mean Squared Error (MSE): the mean squared deviation of all true y values from the line"
                        ),
                        html.Li(
                            "Root Mean Squared Error (RMSE): the square root of the mean squared deviation of all true y values from the line"
                        ),
                        html.Li(
                            "Mean Absolute Error (MAE): the mean absolute deviation of all true y values from the line"
                        ),
                    ]
                ),
                html.P(
                    f"""
    At this point the residuals slider provides a suitable visualization. As soon as you activate the slider you see the distance between each true y value and the line. This deviation is also called residual. If you change ${theta(sub=0)}$ and ${theta(sub=1)}$, the residuals grow and shrink, respectively, and the resulting cost functions improve or become worse. You can observe that after you press the "BEST FIT" button the residuals are as small as possible and MSE, RMSE and MAE take their best possible values. Any further changes to the ${theta(sub=0)}$ and ${theta(sub=1)}$ sliders will increase the residuals and worsen the fit of the line.
    """
                ),
                html.P(
                    """
    In contrast to MSE, RMSE and MAE, it is not true for $R^2$ that smaller values are better. $R^2$ describes which part of the variance is described by the model. R-squared is therefore a value between 0 and 1 for which larger values are better. Thus, an $R^2$ value of 1 means that the model can explain any noise in the data. Very high $R^2$ values should be viewed skeptically as they are an indication of overfitting, which will be discussed in more detail later. We will discuss overfitting in more detail at a later time. Don't be surprised if $R^2$ takes a negative value in one of the following dashboards. $R^2<0$ doesn't indicate that the model is wrong, but rather that the model is extremely poor. This is because a negative $R^2$ implies that the model is not capable of explaining more variance than a horizontal line.
    """
                ),
                html.P(
                    [
                        f"For above's dataset, the best possible line is obtained with ",
                        html.Span(id="inline_theta0_linear"),
                        " and ",
                        html.Span(id="inline_theta1_linear"),
                        ". There are several ways to determine these values. Probably one of the most common approaches is gradient descent, which is a generic optimization algorithm that is the best choice for solving a variety of problems. ",
                        "However, since gradient descent is quite extensive and definitely worth a separate dashboard, I'm going to introduce an alternative approach to find the best possible straight line: the normal equation.",
                    ]
                ),
                html.P(
                    f"""
    The normal equation, unlike gradient descent which is an iterative process, determines the best possible model parameters using a mathematical equation: $$ {hat(theta())} = {subsup(f"({subsup('X',sup='T')}X)",sup=-1)}{subsup('X',sup='T')}y $$
    ${hat(theta())}$ is the value of ${theta()}$ for which the cost function is minimized, $X$ is a matrix containing all values of the descriptive variable, and $y$ is a vector containing the dependent variable. Thus, $X$, $y$ and some algebra is all we need to find the best possible model parameters ${hat(theta())}$. Calculating ${hat(theta())}$ by hand looks like this:
    """
                ),
                normal_equation,
                html.P(
                    f"""
    😨 no one has time for that. Plus, all the computational steps (transposing, inverting, multiplying) are tedious and error-prone. This is why instead we let Python do the hard work for us. Here is the same calculation as above, only much faster since we use Numpy:
    """
                ),
                code_snippet,
                html.P(
                    """
    Both, the code example and the matrix notation, are dynamic and adapt to the dataset used in the dashboard above 😎. Give it a try: generate a new dataset, click on "BEST FIT" and check if the results of the code and matrix notation match the values in the dashboard.
    """
                ),
                html.P(
                    """
    In general, that's all you need to know about univariate linear regression. For the multivariate case, that is, when multiple explanatory variables enter the equation, the approach is very similar. In the next section you will see that although the equations and matrices are larger and more comprehensive, the underlying procedure is the same.
    """
                ),
            ],
            className="bs-component",
        ),
    ]
)
