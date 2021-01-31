import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import dash
import dash_html_components as html

from dash.dependencies import Input, Output, State
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from app import app
from util import array2matrix


@app.callback([Output("sigmoid_graph", "figure")], [Input("dummy", "children")])
def sigmoud_graph(dummy):
    sigmoid = lambda x: (1 / (1 + np.exp(-x)))
    x = np.linspace(-10, 10, 100)
    fig = go.Figure(
        layout=dict(
            yaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[-0.1, 1.1],
            ),
            xaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[-10, 10],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=0, r=0, l=0, b=0),
        )
    )
    fig.add_trace(go.Scatter(x=x, y=sigmoid(x)))

    fig.update_layout(
        xaxis_title="t",
    )

    return [fig]


@app.callback(
    [
        Output("graph_logreg", "figure"),
        Output("confusion_matrix", "children"),
        Output("metrics", "children"),
    ],
    [
        Input("theta0_slider_logreg", "value"),
        Input("theta1_slider_logreg", "value"),
        State("values_logreg", "children"),
    ],
)
def graph_logreg(theta0, theta1, data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]

    sigma = lambda t: 1 / (1 + np.exp(-(t)))
    theta = np.array([[theta0], [theta1]])

    y_proba_new = sigma(X_new_b.dot(theta))
    y_proba = sigma(X_b.dot(theta)) >= 0.5

    try:
        decision_boundary = X_new[y_proba_new >= (1 - y_proba_new)][0]
    except IndexError:
        decision_boundary = 10

    tn = X[(y_proba.ravel() == 0) & (y == 0)]
    fp = X[(y_proba.ravel() == 1) & (y == 0)]
    fn = X[(y_proba.ravel() == 0) & (y == 1)]
    tp = X[(y_proba.ravel() == 1) & (y == 1)]

    fig = go.Figure(
        layout=dict(
            yaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[-0.1, 1.1],
            ),
            xaxis=dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                range=[0, 3],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=0, r=0, l=0, b=0),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_new.ravel(),
            y=y_proba_new.ravel(),
            name="Iris Virginica",
            mode="lines",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_new.ravel(),
            y=1 - y_proba_new.ravel(),
            name="Not Iris Virginica",
            mode="lines",
            line=dict(color="blue", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[decision_boundary, decision_boundary],
            y=[-1, 2],
            name="Decision boundary",
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tp,
            y=np.full((1, len(tp)), 1).ravel(),
            name="True Positives",
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fp,
            y=np.full((1, len(fp)), 0.95).ravel(),
            name="False Positives",
            mode="markers",
            marker=dict(symbol="circle", size=10, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tn,
            y=np.full((1, len(tn)), 0).ravel(),
            name="True Negatives",
            mode="markers",
            marker=dict(symbol="circle", size=10, color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fn,
            y=np.full((1, len(fn)), 0.05).ravel(),
            name="False Negatives",
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="red"),
        )
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Petal width (cm)",
        yaxis_title="Probability",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    cm = [
        html.Tr([html.Td(f"TN: {len(tn)}"), html.Td(f"FP: {len(fp)}")]),
        html.Tr([html.Td(f"FN: {len(fn)}"), html.Td(f"TP: {len(tp)}")]),
    ]

    p = precision_score(y, y_proba, zero_division=1)
    r = recall_score(y, y_proba, zero_division=1)
    a = accuracy_score(y, y_proba)
    f1 = f1_score(y, y_proba, zero_division=1)

    metrics = [
        html.Tr([html.Td("Accuracy"), html.Td(f"{a*100:.2f}%")]),
        html.Tr([html.Td("Precision"), html.Td(f"{p*100:.2f}%")]),
        html.Tr([html.Td("Recall"), html.Td(f"{r*100:.2f}%")]),
        html.Tr([html.Td("F1"), html.Td(f"{f1*100:.2f}%")]),
    ]

    return [fig, cm, metrics]


@app.callback(
    [
        Output("theta0_slider_value_logreg", "children"),
        Output("theta1_slider_value_logreg", "children"),
    ],
    [
        Input("theta0_slider_logreg", "value"),
        Input("theta1_slider_logreg", "value"),
    ],
)
def equations_logreg(theta0, theta1):
    return [
        f"$ \\theta_0 = {theta0:.1f} $",
        f"$ \\theta_1 = {theta1:.1f} $",
    ]


@app.callback(
    [Output("theta0_slider_logreg", "value"), Output("theta1_slider_logreg", "value")],
    [
        Input("best_fit_button_logreg", "n_clicks"),
        State("theta0_slider_logreg", "value"),
        State("theta1_slider_logreg", "value"),
        State("values_logreg", "children"),
    ],
)
def best_fit(n_clicks, theta0, theta1, data):
    df = pd.read_json(data)
    X = df["X"].to_numpy()
    y = df["y"].to_numpy()

    ctx = dash.callback_context.triggered
    if "best_fit_button_logreg" in ctx[0]["prop_id"]:
        log_reg = LogisticRegression(solver="liblinear", random_state=42)
        log_reg.fit(X.reshape(-1, 1), y)

        return [log_reg.intercept_[0], log_reg.coef_[0][0]]
    else:
        return [theta0, theta1]
