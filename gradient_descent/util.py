import dash_bootstrap_components as dbc

external_scripts = [
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML",
    "/static/additional.js",
]
external_stylesheets = [
    "https://bootswatch.com/_assets/css/custom.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.5.0/styles/default.min.css",
    dbc.themes.LUX,
    "/static/additional.css",
    "/static/stackoverflow-dark.css",
]