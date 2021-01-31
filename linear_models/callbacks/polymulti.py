import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import dash
import re
import dash_html_components as html

from dash.dependencies import Input, Output, State
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from app import app
from util import array2matrix


@app.callback(
    [
        Output("values_train_polymulti", "children"),
        Output("values_test_polymulti", "children"),
    ],
    [Input("new_dataset_button_polymulti", "n_clicks")],
)
def new_dataset_polymulti(n_clicks):
    m = 50
    neg = random.choice([-1, 1])
    X_train = 6 * np.random.rand(m, 1) - 3
    y_train = (0.5 * X_train ** 2 + X_train + 2 + np.random.randn(m, 1)) * neg
    X_test = 6 * np.random.rand(m, 1) - 3
    y_test = (0.5 * X_test ** 2 + X_test + 2 + np.random.randn(m, 1)) * neg

    X_train, y_train = X_train.ravel(), y_train.ravel()
    X_test, y_test = X_test.ravel(), y_test.ravel()

    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=["X", "y"])
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=["X", "y"])

    return [df_train.to_json(), df_test.to_json()]


@app.callback(
    [
        Output("graph_train_polymulti", "figure"),
        Output("graph_test_polymulti", "figure"),
    ],
    [
        Input("degree_slider_polymulti", "value"),
        Input("values_train_polymulti", "children"),
        Input("values_test_polymulti", "children"),
        Input("residual_switch_polymulti", "value"),
    ],
)
def graph_polymulti(degree, data_train, data_test, residual_switch):
    df_train = pd.read_json(data_train).sort_values("X")
    df_test = pd.read_json(data_test).sort_values("X")

    X_train = df_train["X"].to_numpy()
    y_train = df_train["y"].to_numpy()
    X_test = df_test["X"].to_numpy()
    y_test = df_test["y"].to_numpy()

    polynomial_regression = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression()),
        ]
    )
    polynomial_regression.fit(X_train.reshape(-1, 1), y_train)

    y_train_pred = polynomial_regression.predict(
        X_train.reshape(-1, 1)
    )  # X_train.reshape(-1,1))
    y_test_pred = polynomial_regression.predict(
        X_test.reshape(-1, 1)
    )  # X_train.reshape(-1,1))
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    fig_train = go.Figure(
        layout=dict(
            yaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df_train["y"].min()
                    - (df_train["y"].max() - df_train["y"].min()) * 0.1,
                    df_train["y"].max()
                    + (df_train["y"].max() - df_train["y"].min()) * 0.1,
                ],
            ),
            xaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df_train["X"].min()
                    - (df_train["X"].max() - df_train["X"].min()) * 0.1,
                    df_train["X"].max()
                    + (df_train["X"].max() - df_train["X"].min()) * 0.1,
                ],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=0, r=0, l=0, b=0),
        )
    )
    fig_train.add_trace(
        go.Scatter(
            x=df_train["X"],
            y=df_train["y"],
            name="Data",
            mode="markers",
            marker=dict(color="firebrick"),
        )
    )
    fig_train.add_trace(
        go.Scatter(
            x=df_train["X"],
            y=y_train_pred,
            name="Predictions",
            mode="lines",
            line=dict(color="blue"),
        )
    )

    fig_train.update_layout(
        xaxis_title="x1",
        yaxis_title="y",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(176,196,222,0.9)",
            title_text="<b>Train</b>",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    fig_test = go.Figure(
        layout=dict(
            yaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df_test["y"].min()
                    - (df_test["y"].max() - df_test["y"].min()) * 0.1,
                    df_test["y"].max()
                    + (df_test["y"].max() - df_test["y"].min()) * 0.1,
                ],
            ),
            xaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[
                    df_test["X"].min()
                    - (df_test["X"].max() - df_test["X"].min()) * 0.1,
                    df_test["X"].max()
                    + (df_test["X"].max() - df_test["X"].min()) * 0.1,
                ],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=0, r=0, l=0, b=0),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=df_test["X"],
            y=df_test["y"],
            name="Data",
            mode="markers",
            marker=dict(color="firebrick"),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=df_test["X"],
            y=y_test_pred,
            name="Predictions",
            mode="lines",
            line=dict(color="blue"),
        )
    )

    fig_test.update_layout(
        xaxis_title="x1",
        yaxis_title="y",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(176,196,222,0.9)",
            title_text="<b>Test</b>",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    if residual_switch != []:
        for i in range(0, len(train_residuals)):
            fig_train.add_trace(
                go.Scatter(
                    x=[df_train["X"].iloc[i], df_train["X"].iloc[i]],
                    y=[y_train_pred[i], y_train_pred[i] + train_residuals[i]],
                    line=dict(color="firebrick", width=1, dash="dash"),
                    showlegend=False,
                )
            )
            fig_test.add_trace(
                go.Scatter(
                    x=[df_test["X"].iloc[i], df_test["X"].iloc[i]],
                    y=[y_test_pred[i], y_test_pred[i] + test_residuals[i]],
                    line=dict(color="firebrick", width=1, dash="dash"),
                    showlegend=False,
                )
            )
        fig_train.add_trace(
            go.Scatter(
                x=[-100, -100],
                y=[-100, -100],
                name="Residuals",
                line=dict(color="firebrick", width=1, dash="dash"),
            )
        )
        fig_test.add_trace(
            go.Scatter(
                x=[-100, -100],
                y=[-100, -100],
                name="Residuals",
                line=dict(color="firebrick", width=1, dash="dash"),
            )
        )

    return [fig_train, fig_test]


@app.callback(
    [
        Output("mse_train_polymulti", "children"),
        Output("rmse_train_polymulti", "children"),
        Output("mae_train_polymulti", "children"),
        Output("r2_train_polymulti", "children"),
        Output("mse_test_polymulti", "children"),
        Output("rmse_test_polymulti", "children"),
        Output("mae_test_polymulti", "children"),
        Output("r2_test_polymulti", "children"),
    ],
    [
        Input("degree_slider_polymulti", "value"),
        Input("values_train_polymulti", "children"),
        Input("values_test_polymulti", "children"),
    ],
)
def metrics_polymulti(degree, data_train, data_test):
    df_train = pd.read_json(data_train).sort_values("X")
    df_test = pd.read_json(data_test).sort_values("X")

    X_train = df_train["X"].to_numpy()
    y_train = df_train["y"].to_numpy()
    X_test = df_test["X"].to_numpy()
    y_test = df_test["y"].to_numpy()

    polynomial_regression = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression()),
        ]
    )
    polynomial_regression.fit(X_train.reshape(-1, 1), y_train)

    y_train_pred = polynomial_regression.predict(X_train.reshape(-1, 1))
    y_test_pred = polynomial_regression.predict(X_test.reshape(-1, 1))

    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    textrm_mse = "\\textrm{MSE}"
    textrm_rmse = "\\textrm{RMSE}"
    textrm_mae = "\\textrm{MAE}"

    return [
        f"$ {mse_train:.2f} $",
        f"$ {rmse_train:.2f} $",
        f"$ {mae_train:.2f} $",
        f"$ {r2_train:.2f} $",
        f"$ {mse_test:.2f} $",
        f"$ {rmse_test:.2f} $",
        f"$ {mae_test:.2f} $",
        f"$ {r2_test:.2f} $",
    ]


@app.callback(
    [
        Output("degree_slider_value_polymulti", "children"),
    ],
    [
        Input("degree_slider_polymulti", "value"),
    ],
)
def slider_values_polymulti(degree):
    return [f"$ \\textrm{{degree}} = {degree:.0f}$"]


@app.callback(
    Output("degree_slider_polymulti", "value"),
    [
        Input("degree1_href", "n_clicks"),
        Input("degree2_href", "n_clicks"),
        Input("degree3_href", "n_clicks"),
        Input("degree4_href", "n_clicks"),
        Input("degree5_href", "n_clicks"),
        Input("degree20_href", "n_clicks"),
        Input("degree25_href", "n_clicks"),
        Input("degree30_href", "n_clicks"),
    ],
)
def hyperlink_update(d1, d2, d3, d4, d5, d10, d20, d30):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if changed_id == ".":
        return 1
    else:
        degree = int(re.findall(r"\d+", changed_id)[0])
        return degree


@app.callback(
    [
        Output("table_degree1", "children"),
        Output("table_degree2", "children"),
        Output("table_degree20", "children"),
    ],
    [
        Input("values_train_polymulti", "children"),
        Input("values_test_polymulti", "children"),
    ],
)
def update_ul(data_train, data_test):
    df_train = pd.read_json(data_train).sort_values("X")
    df_test = pd.read_json(data_test).sort_values("X")

    X_train = df_train["X"].to_numpy()
    y_train = df_train["y"].to_numpy()
    X_test = df_test["X"].to_numpy()
    y_test = df_test["y"].to_numpy()

    ret = []
    for degree in [1, 2, 20]:
        polynomial_regression = Pipeline(
            [
                (
                    "poly_features",
                    PolynomialFeatures(degree=degree, include_bias=False),
                ),
                ("std_scaler", StandardScaler()),
                ("lin_reg", LinearRegression()),
            ]
        )
        polynomial_regression.fit(X_train.reshape(-1, 1), y_train)

        y_train_pred = polynomial_regression.predict(X_train.reshape(-1, 1))
        y_test_pred = polynomial_regression.predict(X_test.reshape(-1, 1))

        mse_train = mean_squared_error(y_train, y_train_pred)
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        children = [
            html.Tr([html.Td(), html.Td("Train"), html.Td("Test")]),
            html.Tr(
                [
                    html.Td(f"MSE"),
                    html.Td(f"${mse_train:.2f}$"),
                    html.Td(f"${mse_test:.2f}$"),
                ]
            ),
            html.Tr(
                [
                    html.Td(f"RMSE"),
                    html.Td(f"${rmse_train:.2f}$"),
                    html.Td(f"${rmse_test:.2f}$"),
                ]
            ),
            html.Tr(
                [
                    html.Td(f"MAE"),
                    html.Td(f"${mae_train:.2f}$"),
                    html.Td(f"${mae_test:.2f}$"),
                ]
            ),
            html.Tr(
                [
                    html.Td(f"$R^2$"),
                    html.Td(f"${r2_train:.2f}$"),
                    html.Td(f"${r2_test:.2f}$"),
                ]
            ),
        ]

        ret.append(children)

    return ret
