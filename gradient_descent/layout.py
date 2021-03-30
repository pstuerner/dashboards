import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go

from util import X,y,m,theta0_best,theta1_best
from layouts import (
    introduction,
    section1_layout,
    section2_layout,
    section3_layout,
    section4_layout,
    example3_layout,
    example4_layout,
    toc,
    wrapup
)

layout = dbc.Jumbotron(
    [
        introduction,
        toc,
        section1_layout,
        section2_layout,
        section3_layout,
        section4_layout,
        wrapup,

        html.Div(pd.DataFrame(np.c_[np.ones(len(X)),X,y],columns=['b','X','y']).to_json(orient='split'), id='data', style={'display':'none'}),
        html.Div(pd.DataFrame(np.c_[np.ones(len(X)),(X - X.mean()) / X.std(),y],columns=['b','X','y']).to_json(orient='split'), id='data_scaled', style={'display':'none'}),
        html.Div([theta0_best,theta1_best], id='best_theta', style={'display':'none'}),
        html.Div(id='data_racetrack', style={'display':'none'}),
        html.Div(id='data_contour', style={'display':'none'}),
        html.Div(id='best_theta_racetrack', style={'display':'none'}),
        html.Div(id='racetrack_time', style={'display':'none'}),
        html.Div([], id='bgd_history', style={'display':'none'}),
        html.Div([], id='sgd_history', style={'display':'none'}),
        html.Div([], id='mbgd_history', style={'display':'none'}),
    ],
    className="col-12",
)