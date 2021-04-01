import dash_core_components as dcc
import dash_html_components as html

from app import app
from layout import layout
from callbacks import section1, section2, section4

if __name__ == "__main__":
    app.layout = layout
    app.run_server(debug=True, host='192.168.0.199')
