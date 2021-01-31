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
                        html.Td(""),
                        html.Th("MSE"),
                        html.Th("RMSE"),
                        html.Th("MAE"),
                        html.Th("$R^2$"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Train"),
                        html.Td(id="mse_train_polymulti"),
                        html.Td(id="rmse_train_polymulti"),
                        html.Td(id="mae_train_polymulti"),
                        html.Td(id="r2_train_polymulti"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Test"),
                        html.Td(id="mse_test_polymulti"),
                        html.Td(id="rmse_test_polymulti"),
                        html.Td(id="mae_test_polymulti"),
                        html.Td(id="r2_test_polymulti"),
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
                            id="degree_slider_value_polymulti",
                            style={
                                "width": "20%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="degree_slider_polymulti",
                                min=1,
                                max=30,
                                step=1,
                                value=1,
                                updatemode="drag",
                                className="custom-range",
                            )
                        ),
                    ]
                ),
            ],
            className="no-padding",
            borderless=True,
            style={"table-layout": "fixed"},
        ),
        html.Div(
            id="values_train_polymulti",
            style={"display": "none"},
        ),
        html.Div(
            id="values_test_polymulti",
            style={"display": "none"},
        ),
    ],
    className="col-lg-4 col-md-4 col-sm-12 pb-3",
    style={
        "margin-left": "auto",
        "margin-right": "auto",
        "text-align": "center",
    },
)

graph_train = html.Div(
    [
        dcc.Graph(
            id="graph_train_polymulti",
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
    className="col-lg-5 col-md-5 col-sm-12 pb-3",
)

graph_test = html.Div(
    [
        dcc.Graph(
            id="graph_test_polymulti",
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
    className="col-lg-5 col-md-5 col-sm-12 pb-3",
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
                    id="residual_switch_polymulti",
                    switch=True,
                    inline=True,
                    style={"padding-bottom": "10px"},
                ),
                dbc.Button(
                    "New Dataset",
                    id="new_dataset_button_polymulti",
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

layout = html.Div(
    [
        html.Div(
            [
                html.H2("Overfitting", className="display-5"),
                html.P(
                    """
            Machine learning is full of tradeoffs, they are around every corner and you can hardly make a decision without balancing your options. But overfitting/underfitting is definitely one of the best-known tradeoffs in machine learning. Yet it happens all the time and every time I catch myself doing it I feel a sense of shame 💩. Overfitting is so obvious yet so tempting.
            """
                ),
                html.P(
                    """
            You can explore the topic yourself in the following dashboard. The slider changes the degrees of freedom of a polynomial regression, which is trained using the training dataset (scatter plot in the left figure). Next, we use the trained model to make predictions on the test dataset in the right figure. Play around with the slider and observe how the curve behaves in both figures. The training and test metrics provide interesting insights into the cross-dataset performance of the model.
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
                    ]
                ),
                dbc.Row(
                    [
                        graph_train,
                        graph_test,
                        graph_buttons,
                    ],
                    style={"padding-top": "25px"},
                ),
            ],
            className="bs-component",
        ),
        html.Div(
            [
                html.P(
                    [
                        """
            As mentioned earlier, linear regression is a high bias and low variance model. Simply put, linear regression has little flexibility because it is given that the dataset should be explainable in a linear fashion. Polynomial features are a way to reduce the bias of a linear regression and at the same time increase the model's variance. In other words, the model becomes more flexible. Let's figure out what this looks like while changing the degrees of freedom.
            Note: you can click on the hyperlinks in the text to see each scenario in the dashboard above.
            """
                    ]
                ),
                html.P(
                    [
                        " For ",
                        html.A(
                            "degree = 1",
                            href="javascript:void(0);",
                            id="degree1_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        " we get a straight line, which makes sense since this is just a linear regression. The train and test metrics are quite bad and fairly similar:",
                    ]
                ),
                html.Div(
                    dbc.Table(
                        id="table_degree1",
                        bordered=False,
                        style={
                            "width": "1%",
                            "margin-left": "auto",
                            "margin-right": "auto",
                            "padding": "0.75rem",
                        },
                    )
                ),
                html.P(
                    [
                        "Similarly bad metrics on both train and test are a good indication of underfitting. Our model is too simple, or in other words, we model both datasets equally poorly.  This is obvious, since we are trying to approximate a quadratic scatterplot using a linear function. "
                        "If we increase the degrees of freedom to ",
                        html.A(
                            "degree = 2",
                            href="javascript:void(0);",
                            id="degree2_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        ", the model performance improves dramatically — on both datasets! The train and test metrics now look more familiar:",
                    ]
                ),
                html.Div(
                    dbc.Table(
                        id="table_degree2",
                        bordered=False,
                        style={
                            "width": "1%",
                            "margin-left": "auto",
                            "margin-right": "auto",
                            "padding": "0.75rem",
                        },
                    )
                ),
                html.P(
                    [
                        "Better metrics on train than test are generally a good thing.  It speaks against underfitting, however we cannot exclude that our model is already overtrained. ",
                        "If we increase the degrees of freedom to ",
                        html.A(
                            "degree = 3",
                            href="javascript:void(0);",
                            id="degree3_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        ", ",
                        html.A(
                            "degree = 4",
                            href="javascript:void(0);",
                            id="degree4_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        " or ",
                        html.A(
                            "degree = 5",
                            href="javascript:void(0);",
                            id="degree5_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        " we see minimal to no improvement in all metrics. This is a strong indication that we are close to overfitting and will only continue to make things worse. Let’s increase the degrees of freedom to ",
                        html.A(
                            "degree = 20",
                            href="javascript:void(0);",
                            id="degree20_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        " and observe what happens to the performance metrics:",
                    ]
                ),
                html.Div(
                    dbc.Table(
                        id="table_degree20",
                        bordered=False,
                        style={
                            "width": "1%",
                            "margin-left": "auto",
                            "margin-right": "auto",
                            "padding": "0.75rem",
                        },
                    )
                ),
                html.P(
                    [
                        "Holy mother of machine learning 😱! We improved our train performance again... while totally messing up test performance at the same time. That's overfitting at its best. We can continue this effect by increasing the degrees of freedom to ",
                        html.A(
                            "degree = 25",
                            href="javascript:void(0);",
                            id="degree25_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        " or ",
                        html.A(
                            "degree = 30",
                            href="javascript:void(0);",
                            id="degree30_href",
                            style={
                                "color": "blue",
                                "font-weight": "bold",
                            },
                        ),
                        ". The training performance will continuously improve, while our predictions on unseen data will get worse and worse.",
                    ]
                ),
                html.P(
                    """
            Now you should have a good overview of how linear regression models work under the hood as well as what their strengths and pitfalls are. I would like to go into more detail on another variation of linear models, once again highlighting how diverse this family of algorithms is.
            """
                ),
            ],
            className="bs-component",
        ),
    ]
)
