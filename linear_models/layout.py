import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go

from util import df_iris, hat, theta, subsup
from layouts import (
    introduction_layout,
    linear_layout,
    poly_layout,
    polymulti_layout,
    logreg_layout,
    conclusion_layout,
)

layout = dbc.Jumbotron(
    [
        introduction_layout,
        linear_layout,
        poly_layout,
        polymulti_layout,
        logreg_layout,
        conclusion_layout,
    ],
    className="col-12",
    style={"padding": "2rem 2rem"},
)
