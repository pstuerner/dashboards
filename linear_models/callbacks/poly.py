import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import dash
import dash_core_components as dcc

from dash.dependencies import Input, Output, State
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app import app
from util import array2matrix, hat, theta


@app.callback(
    [Output("graph_poly", "figure")],
    [
        Input("theta0_slider_poly", "value"),
        Input("theta1_slider_poly", "value"),
        Input("theta2_slider_poly", "value"),
        Input("prediction_slider_poly", "value"),
        Input("residual_switch_poly", "value"),
        Input("values_poly", "children"),
    ],
)
def graph_poly(theta0, theta1, theta2, prediction, residual_switch, data):
    df = pd.read_json(data)
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
        df["X"].to_numpy().reshape(-1, 1)
    )
    l = theta1 * X_poly[:, 0] + theta2 * X_poly[:, 1] + theta0
    residuals = df["y"] - l
    theta = np.array([[theta0], [theta1], [theta2]])
    x = np.array([[1], [prediction], [prediction ** 2]])
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
    [Output("values_poly", "children")], [Input("new_dataset_button_poly", "n_clicks")]
)
def new_dataset_poly(n_clicks):
    m = 25
    X = np.sort(6 * np.random.rand(m, 1) - 3, axis=0)
    y = (
        random.randint(10, 250) / 100 * X ** 2
        + X
        + random.randint(0, 20)
        + np.random.randn(m, 1)
    ) * random.choice([-1, 1])

    X = X.ravel()
    y = y.ravel()
    df = pd.DataFrame(np.c_[X, y], columns=["X", "y"])

    return [df.to_json()]


@app.callback(
    [
        Output("theta0_slider_poly", "value"),
        Output("theta1_slider_poly", "value"),
        Output("theta2_slider_poly", "value"),
        Output("prediction_slider_poly", "value"),
        Output("theta0_slider_poly", "min"),
        Output("theta1_slider_poly", "min"),
        Output("theta2_slider_poly", "min"),
        Output("prediction_slider_poly", "min"),
        Output("theta0_slider_poly", "max"),
        Output("theta1_slider_poly", "max"),
        Output("theta2_slider_poly", "max"),
        Output("prediction_slider_poly", "max"),
    ],
    [Input("values_poly", "children"), Input("best_fit_button_poly", "n_clicks")],
)
def sliders_poly(data, n_clicks):
    ctx = dash.callback_context.triggered
    df = pd.read_json(data)

    if "values_poly" in ctx[0]["prop_id"]:
        return [
            df["y"].mean(),
            0,
            0,
            0,
            df["y"].min() - (df["y"].max() - df["y"].min()) * 0.1,
            -10,
            -5,
            df["X"].min(),
            df["y"].max() + (df["y"].max() - df["y"].min()) * 0.1,
            10,
            5,
            df["X"].max(),
        ]
    else:
        X = df["X"].to_numpy()
        y = df["y"].to_numpy()
        X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
            df["X"].to_numpy().reshape(-1, 1)
        )
        X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        theta = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)
        return [
            theta[0],
            theta[1],
            theta[2],
            0,
            df["y"].min() - (df["y"].max() - df["y"].min()) * 0.1,
            -10,
            -5,
            df["X"].min(),
            df["y"].max() + (df["y"].max() - df["y"].min()) * 0.1,
            10,
            5,
            df["X"].max(),
        ]


@app.callback(
    [Output("normal_equation_poly", "children")], [Input("values_poly", "children")]
)
def normal_equation_poly(data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
        df["X"].to_numpy().reshape(-1, 1)
    )
    X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
    cdot = "\\cdot"
    h1 = "{T}"
    h2 = "{-1}"

    return [
        f"$$ {hat(theta())} = (X^{h1}X)^{h2}\\cdot X^{h1}y $$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_poly_b)}^{h1}\\cdot{array2matrix(X_poly_b)}\\right)^{h2}\\cdot{array2matrix(X_poly_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_poly_b.T)}\\cdot{array2matrix(X_poly_b)}\\right)^{h2}\\cdot{array2matrix(X_poly_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = \\left({array2matrix(X_poly_b.T.dot(X_poly_b))}\\right)^{h2}\\cdot{array2matrix(X_poly_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = {array2matrix(np.linalg.inv(X_poly_b.T.dot(X_poly_b)))}\\cdot{array2matrix(X_poly_b)}\\cdot{array2matrix(y.reshape(-1,1))}$$"
        f"$$ {hat(theta())} = {array2matrix(np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y.reshape(-1,1)))} $$"
    ]


@app.callback(
    [
        Output("theta0_slider_value_poly", "children"),
        Output("theta1_slider_value_poly", "children"),
        Output("theta2_slider_value_poly", "children"),
        Output("prediction_slider_value_poly", "children"),
        Output("equation_poly", "children"),
    ],
    [
        Input("theta0_slider_poly", "value"),
        Input("theta1_slider_poly", "value"),
        Input("theta2_slider_poly", "value"),
        Input("prediction_slider_poly", "value"),
    ],
)
def equations_poly(theta0, theta1, theta2, prediction):
    h1 = "\\hat{y}"
    theta = np.array([[theta0], [theta1], [theta2]])
    x = np.array([[1], [prediction], [prediction ** 2]])
    y_pred = theta.T.dot(x)

    return [
        f"$ \\theta_0 = {theta0:.2f} $",
        f"$ \\theta_1 = {theta1:.2f} $",
        f"$ \\theta_2 = {theta2:.2f} $",
        f"$ x_1 = {prediction:.2f} $",
        f"$ {h1} = \\theta^T x = {array2matrix(theta.T)} {array2matrix(x)} = {y_pred[0][0]:.2f} $",
    ]


@app.callback(
    [
        Output("mse_poly", "children"),
        Output("rmse_poly", "children"),
        Output("mae_poly", "children"),
        Output("r2_poly", "children"),
    ],
    [
        Input("theta0_slider_poly", "value"),
        Input("theta1_slider_poly", "value"),
        Input("theta2_slider_poly", "value"),
        Input("values_poly", "children"),
    ],
)
def metrics_poly(theta0, theta1, theta2, data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
        X.reshape(-1, 1)
    )
    l = theta0 + theta1 * X_poly[:, 0] + theta2 * X_poly[:, 1]

    mse = mean_squared_error(y, l)
    rmse = mean_squared_error(y, l, squared=False)
    mae = mean_absolute_error(y, l)
    r2 = r2_score(y, l)

    textrm_mse = "\\textrm{MSE}"
    textrm_rmse = "\\textrm{RMSE}"
    textrm_mae = "\\textrm{MAE}"

    return [f"$ {mse:.2f} $", f"$ {rmse:.2f} $", f"$ {mae:.2f} $", f"$ {r2:.2f} $"]


@app.callback(
    [Output("code_example_poly", "children")], [Input("values_poly", "children")]
)
def code_example_poly(data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
        X.reshape(-1, 1)
    )
    X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
    theta_best = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)

    code = dcc.Markdown(
        [
            f"""
        ```python
        >>> import numpy as np
        >>> from sklearn.preprocessing import PolynomialFeatures
        >>> X=np.array([{','.join(X.round(2).astype(str))}])
        >>> y=np.array([{','.join(y.round(2).astype(str))}])
        >>> X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X.reshape(-1,1))
        >>> X_poly_b = np.c_[np.ones((len(X_poly), 1)), X_poly]
        >>> theta = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)
        >>> theta
        array([{theta_best[0]:.2f}, {theta_best[1]:.2f}, {theta_best[2]:.2f}])
        ```
        """
        ]
    )

    return [code]
