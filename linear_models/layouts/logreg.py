import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from util import df_iris, hat, theta, subsup

sigmoid_graph = html.Div(
    [
        html.Div(
            dcc.Graph(
                id="sigmoid_graph",
                config={"staticPlot": True},
                style={"height": "300px"},
            )
        )
    ],
    className="col-lg-6 col-mg-6 col-sm-12 mb-3 mt-3",
    id="dummy",
    style={"margin-left": "auto", "margin-right": "auto"},
)

code_snippet = html.Div(
    [
        dcc.Markdown(
            [
                f"""
                ```python
                >>> import numpy as np
                >>> from sklearn import datasets
                >>> from sklearn.linear_model import LogisticRegression
                >>> iris = datasets.load_iris()
                >>> X = iris["data"][:, 3:]
                >>> y = (iris["target"] == 2).astype(np.int)
                >>> X.ravel()
                array([0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, ...])
                >>> y.ravel()
                array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...])
                ```
                """
            ]
        ),
    ],
    className="col-lg-6 col-md-8 col-sm-12",
    style={"margin-left": "auto", "margin-right": "auto"},
)

metrics = html.Div(
    [
        html.H4("Metrics", style={"margin-bottom": "1.5em"}),
        html.Div(
            [
                html.Div(
                    [
                        dbc.Table(
                            id="confusion_matrix",
                            bordered=True,
                            style={"align-self": "center"},
                        )
                    ],
                    className="col-lg-6 col-md-6 col-sm-12",
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "margin-left": "auto",
                        "margin-right": "auto",
                    },
                ),
                html.Div(
                    [dbc.Table(id="metrics", bordered=False)],
                    className="col-lg-6 col-md-6 col-sm-12",
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "text-align": "center",
                    },
                ),
            ],
            style={"display": "flex", "flex-wrap": "wrap"},
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
                            id="theta0_slider_value_logreg",
                            style={
                                "width": "20%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta0_slider_logreg",
                                min=-8,
                                max=1,
                                step=0.1,
                                value=-3,
                                updatemode="drag",
                            )
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            id="theta1_slider_value_logreg",
                            style={
                                "width": "20%",
                                "text-align": "left",
                            },
                        ),
                        html.Td(
                            dcc.Slider(
                                id="theta1_slider_logreg",
                                min=0,
                                max=10,
                                step=0.1,
                                value=3,
                                updatemode="drag",
                            )
                        ),
                    ]
                ),
            ],
            className="no-padding",
            borderless=True,
            style={"table-layout": "fixed"},
        ),
        dbc.Button(
            "Best fit",
            id="best_fit_button_logreg",
            className="btn btn-outline-primary",
        ),
        html.Div(
            df_iris.to_json(),
            id="values_logreg",
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

graph = dbc.Col(
    [
        dcc.Graph(
            id="graph_logreg",
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
    width=12,
)

layout = html.Div(
    [
        html.Div(
            [
                html.H2("Logistic Regression", className="display-5"),
                html.P(
                    """
            In contrast to linear or polynomial regression, logistic regression doesn't predict a continous value (e.g. stock price, temparature, height), but rather predicts a discrete value like whether or not an instance belongs to a certain class. This is called classification and, along with regression, is one of the most common problems in machine learning. In the following, we explore a binary classification problem in greater detail. Binary classification problems deal with the question whether an instance belongs to the positive (1) or negative (0) class. For example, an algorithm that predicts whether a flower is a sunflower (positive class, 1) or a rose (negative class 0).
            """
                ),
                html.P(
                    """
            In general, however, things don't change that much. A logistic regression is also a linear model that makes a prediction based on the weighted sum of some input features (plus a bias term). The main difference is that the target variable we want to predict is between 0 and 1. The reason for this is simply that we are predicting probabilities and since probabilities are constrained to be in the range of 0 to 1 any other values wouldn't really make sense. Simply put, the linear regression result is squeezed into a corridor between zero and one. To achieve this, the logistic results of the linear regression are calculated. The logistic is a sigmoid function which is defined as follows: $$ \\sigma(t) = \\frac{1}{1+e^{-t}} $$
            """
                ),
                sigmoid_graph,
                html.P(
                    """
            $\\sigma(t)$ has a distinct S-shape and only outputs values between zero and one. This means that $\\hat{y} = \\theta^{T}x$ becomes: $$ \\hat{p} = \\sigma(\\theta^{T}x) $$
            Determining whether an instance belongs to the positive or negative class is really easy just by looking at $\\hat{p}$:
            $$ \\hat{y} = \\begin{cases} 0 & \\text{if } \\hat{p} < 0.5 \\\\ 1 & \\text{if } \\hat{p} \\geq 0.5 \\end{cases} $$
            Simple, right? The final prediction only depends on whether the probability $\\hat{p}$ is greater than or equal to 0.5. If $\\hat{p}$ is greater than or equal to 0.5 then $\\hat{y}=1$, otherwise $\\hat{y}=0$. 
            """
                ),
                html.P(
                    [
                        "As with linear regression, the perfect curve is found by minimizing a cost function. However, the previously used cost function of linear regression does not work in this case just because ",
                        html.A(
                            "one does not simply regress a binary outcome applying ordinary least squares",
                            href="https://i.imgur.com/GOhfk6k.png",
                            target="_blank",
                        ),
                        ". The cost function for logistic regression looks like this: $$ -\\frac{1}{m} \\sum_{i=1}^{m}[y^{(i)}\\log{(\\hat{p}^{(i)})}+(1-\\hat{y}^{i})\\log{(1-\\hat{p}^{(i)})}] $$",
                    ]
                ),
                html.P(
                    [
                        """
            I don't want to go into too much detail, but the most important insight is that wrong predictions are penalized while correct ones are rewarded. This is because $y^{(i)}\\log{(\\hat{p}^{(i)})}$ is large (bad) if the instance belongs to the positive class, but the model predicts a low probability. Similarly, for instances belonging to the negative class, $(1-\\hat{y}^{i})\\log{(1-\\hat{p}^{(i)})}$ becomes large (bad) if a high probability is predicted. Unfortunately, there is no known closed-form equation for the above cost function, which means that we cannot determine the perfect model parameters like above when we used $\\hat{\\theta}=(X^{T}X)^{-1}X^{T}y$. The good news is that the cost function is convex and gradient descent is therefore guaranteed to reach the global minimum. Enough theory, let's construct a small example to understand the subject.
            """
                    ]
                ),
                code_snippet,
                html.P(
                    [
                        "We use sklearn's ",
                        html.A(
                            html.Code("datasets.load_iris()"),
                            href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
                            target="_blank",
                        ),
                        " method to load the iris dataset, which is a common and simple dataset for mulit-class classification of a particular plant species. The dataset contains three different species of irises (setosa, versicolour, and virginica), as well as their petal and sepal length. Basically, it is a dataset of three flowers of the same species (labels) with different measurements (features). Since we only want to train a binary classifier, we restrict the target variable to iris virginica: ",
                        html.Code('y = (iris["target"] == 2).astype(np.int)'),
                        ". This means that all iris virginicas have the y-value 1, all others 0. In order to visualize the model as clearly as possible, we also restrict the number of features and only select the petal length: ",
                        html.Code('X = iris["data"][:, 3:]'),
                        ". Thus, we train a univariate binary classifier that predicts whether an iris is an iris virginica or not based on the petal length.",
                    ]
                ),
                html.P(
                    """
            In the following dashboard you can see how the binary classifier behaves and how changes in the model parameters affect the predictions.
            """
                ),
            ],
            className="bs-component",
        ),
        html.Div(
            [
                dbc.Row([metrics, parameters]),
                dbc.Row(
                    [graph],
                    style={"padding-top": "25px"},
                ),
            ],
            className="bs-component",
        ),
        html.Div(
            [
                html.P(
                    """
            The continuously increasing sigmoid curve indicates the probability with which an instance belongs to the iris virginica class. The continuously decreasing sigmoid curve indicates the opposite, namely the probability with which an instance does not belong to the iris virginica class. The decision boundary is always located at the intersection of the two sigmoid curves. Both model parameters shift the decision boundary on the x-axis. While $\\theta_0$ shifts the whole sigmoid curve to the left and right, $\\theta_1$ changes the slope in the area of the decision boundary. Both parameters push the decision boundary to the right for small values and to the left for large values.
            """
                ),
                html.P(
                    [
                        'By clicking on "BEST FIT" you will get the best sigmoid curve that best explains the data. As we already know from the logistic regression cost function, the best way to describe the data is to correctly classify as many positive and negative instances as possible. The confusion matrix and performance metrics help you compare different parameter combinations and determine if the model is getting better or worse. If you don\'t have any prior experience with performance metrics (accuracy, prediction, recall, f1) and the confusion matrix you can read more about it in one of my other dashboards. In ',
                        html.A(
                            '"Unconfusing the Confusion Matrix"',
                            href="https://www.philippstuerner.com/dashboards/confusionmatrix",
                            target="_blank",
                        ),
                        " I describe what performance metrics and the confusion matrix are using a binary classifier that predicts whether a handwritten digit is a five or not.",
                    ]
                ),
            ],
            className="bs-component",
        ),
    ]
)
