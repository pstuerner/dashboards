import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from layouts import (
    introduction,
    section1_layout,
    example2_layout,
    example3_layout,
    example4_layout,
    toc
)

X = np.load('X.npy') #= 2 * np.random.rand(m, 1)
y = np.load('y.npy') #= 4 + 3 * X + np.random.randn(m, 1)
m = len(X)

lr = LinearRegression()
lr.fit(X,y)
theta0_best, theta1_best = lr.intercept_[0], lr.coef_[0][0]

layout = dbc.Jumbotron(
    [
        introduction,
        toc,
        section1_layout,
        example2_layout,
        example4_layout,

        html.Div(pd.DataFrame(np.c_[np.ones(len(X)),X,y],columns=['b','X','y']).to_json(orient='split'), id='data', style={'display':'none'}),
        html.Div(pd.DataFrame(np.c_[np.ones(len(X)),(X - X.mean()) / X.std(),y],columns=['b','X','y']).to_json(orient='split'), id='data_scaled', style={'display':'none'}),
        html.Div([theta0_best,theta1_best], id='best_theta', style={'display':'none'})
    ],
    className="col-12",
    style={"padding": "2rem 2rem"},
)