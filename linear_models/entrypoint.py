import dash_core_components as dcc
import dash_html_components as html

from app import app
from layout import layout
from callbacks import logreg, linear, poly, polymulti
from dash.dependencies import Input, Output

if __name__ == "__main__":
    app.layout = layout
    app.run_server(debug=True)
