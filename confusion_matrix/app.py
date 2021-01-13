import dash
from util import external_scripts, external_stylesheets

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    external_scripts=external_scripts
    )
server = app.server