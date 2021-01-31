import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import dash
import dash_core_components as dcc

from dash.dependencies import Input, Output, State
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app import app
from util import array2matrix, hat, theta


@app.callback(
    [Output("graph_linear", "figure")],
    [
        Input("theta0_slider_linear", "value"),
        Input("theta1_slider_linear", "value"),
        Input("prediction_slider_linear", "value"),
        Input("residual_switch_linear", "value"),
        Input("values_linear", "children"),
    ],
)
def graph_linear(theta0, theta1, prediction, residual_switch, data):
    df = pd.read_json(data)
    l = theta0 + theta1 * df["X"]
    residuals = df["y"] - l
    theta = np.array([[theta0], [theta1]])
    x = np.array([[1], [prediction]])
    y_pred = theta.T.dot(x)[0][0]

    fig = go.Figure(
        layout=dict(
            yaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df["y"].min() - (df["y"].max() - df["y"].min()) * 0.1,
                    df["y"].max() + (df["y"].max() - df["y"].min()) * 0.1,
                ],
            ),
            xaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df["X"].min() - (df["X"].max() - df["X"].min()) * 0.1,
                    df["X"].max() + (df["X"].max() - df["X"].min()) * 0.1,
                ],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=0, r=0, l=0, b=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["X"],
            y=df["y"],
            name="Data",
            mode="markers",
            marker=dict(color="firebrick"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["X"], y=l, name="Predictions", mode="lines", line=dict(color="blue")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[prediction],
            y=[y_pred],
            name="Single Prediction",
            mode="markers",
            marker=dict(size=10, color="green"),
        )
    )

    fig.update_layout(
        xaxis_title="x1",
        yaxis_title="y",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(176,196,222,0.9)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    if residual_switch != []:
        for i, r in enumerate(residuals):
            fig.add_trace(
                go.Scatter(
                    x=[df["X"].iloc[i], df["X"].iloc[i]],
                    y=[l[i], l[i] + r],
                    line=dict(color="firebrick", width=1, dash="dash"),
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[-100, -100],
                y=[-100, -100],
                name="Residuals",
                line=dict(color="firebrick", width=1, dash="dash"),
            )
        )

    return [fig]


@app.callback(
    [Output("values_linear", "children")],
    [Input("new_dataset_button_linear", "n_clicks")],
)
def new_dataset_linear(n_clicks):
    X, y = make_regression(
        n_samples=25,
        n_features=1,
        bias=random.randint(0, 30),
        n_informative=1,
        noise=random.randint(15, 30),
    )
    X = X.ravel()
    y = y.ravel() * random.choice([-1, 1])
    df = pd.DataFrame(np.c_[X, y], columns=["X", "y"])

    return [df.to_json()]


@app.callback(
    [
        Output("theta0_slider_linear", "value"),
        Output("theta1_slider_linear", "value"),
        Output("prediction_slider_linear", "value"),
        Output("theta0_slider_linear", "min"),
        Output("theta1_slider_linear", "min"),
        Output("prediction_slider_linear", "min"),
        Output("theta0_slider_linear", "max"),
        Output("theta1_slider_linear", "max"),
        Output("prediction_slider_linear", "max"),
    ],
    [Input("values_linear", "children"), Input("best_fit_button_linear", "n_clicks")],
)
def slider_linear(data, n_clicks):
    ctx = dash.callback_context.triggered
    df = pd.read_json(data)

    if "values_linear" in ctx[0]["prop_id"]:
        return [
            df["y"].mean(),
            0,
            0,
            df["y"].min() - 0.2,
            df["y"].max() * (-3),
            df["X"].min(),
            df["y"].max(),
            df["y"].max() * 3,
            df["X"].max(),
        ]
    else:
        X = df["X"].to_numpy()
        y = df["y"].to_numpy()
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        return [
            theta[0],
            theta[1],
            0,
            df["y"].min() - 0.2,
            df["y"].max() * (-3),
            df["X"].min(),
            df["y"].max(),
            df["y"].max() * 3,
            df["X"].max(),
        ]


@app.callback(
    [
        Output("theta0_slider_value_linear", "children"),
        Output("theta1_slider_value_linear", "children"),
        Output("prediction_slider_value_linear", "children"),
        Output("equation_linear", "children"),
    ],
    [
        Input("theta0_slider_linear", "value"),
        Input("theta1_slider_linear", "value"),
        Input("prediction_slider_linear", "value"),
    ],
)
def equations_linear(theta0, theta1, prediction):
    h1 = "\\hat{y}"
    theta = np.array([[theta0], [theta1]])
    x = np.array([[1], [prediction]])
    y_pred = theta.T.dot(x)
    return [
        f"$ \\theta_0 = {theta0:.2f} $",
        f"$ \\theta_1 = {theta1:.2f} $",
        f"$ x_1 = {prediction:.2f} $",
        f"$ {h1} = \\theta^T x = {array2matrix(theta.T)} {array2matrix(x)} = {y_pred[0][0]:.2f} $",
    ]


@app.callback(
    [
        Output("mse_linear", "children"),
        Output("rmse_linear", "children"),
        Output("mae_linear", "children"),
        Output("r2_linear", "children"),
    ],
    [
        Input("theta0_slider_linear", "value"),
        Input("theta1_slider_linear", "value"),
        Input("values_linear", "children"),
    ],
)
def metrics_linear(theta0, theta1, data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    l = theta0 + theta1 * X

    mse = mean_squared_error(y, l)
    rmse = mean_squared_error(y, l, squared=False)
    mae = mean_absolute_error(y, l)
    r2 = r2_score(y, l)

    textrm_mse = "\\textrm{MSE}"
    textrm_rmse = "\\textrm{RMSE}"
    textrm_mae = "\\textrm{MAE}"

    return [f"$ {mse:.2f} $", f"$ {rmse:.2f} $", f"$ {mae:.2f} $", f"$ {r2:.2f} $"]


@app.callback(
    [
        Output("inline_theta0_linear", "children"),
        Output("inline_theta1_linear", "children"),
    ],
    [Input("values_linear", "children")],
)
def linear_inline_values(data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return [f"$\\theta_0 = {theta_best[0]:.2f}$", f"$\\theta_1 = {theta_best[1]:.2f}$"]


@app.callback(
    [Output("code_example_linear", "children")], [Input("values_linear", "children")]
)
def code_example_linear(data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    code = dcc.Markdown(
        [
            f"""
        ```python
        >>> import numpy as np
        >>> X=[{','.join(X.round(2).astype(str))}]
        >>> y=[{','.join(y.round(2).astype(str))}]
        >>> X_b = np.c_[np.ones((len(X), 1)), X]
        >>> theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        >>> theta
        array([{theta_best[0]:.2f}, {theta_best[1]:.2f}])
        ```
        """
        ]
    )

    return [code]


@app.callback(
    [Output("normal_equation_linear", "children")], [Input("values_linear", "children")]
)
def normal_equation_linear(data):
    df = pd.read_json(data)

    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    cdot = "\\cdot"
    h1 = "{T}"
    h2 = "{-1}"

    return [
        f"$$ {hat(theta())} = (X^{h1}X)^{h2}\\cdot X^{h1}y $$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_b)}^{h1}\\cdot{array2matrix(X_b)}\\right)^{h2}\\cdot{array2matrix(X_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_b.T)}\\cdot{array2matrix(X_b)}\\right)^{h2}\\cdot{array2matrix(X_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_b.T.dot(X_b))}\\right)^{h2}\\cdot{array2matrix(X_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = {array2matrix(np.linalg.inv(X_b.T.dot(X_b)))}\\cdot{array2matrix(X_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = {array2matrix(np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y.reshape(-1,1)))} $$"
    ]
