import numpy as np
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

m=50
j = lambda X_b,y,theta0,theta1: 1/2*1/m*((X_b.dot(np.array([[theta0],[theta1]]))-y)**2).sum()
djt0 = lambda X_b,y,theta0,theta1: 1/m*(X_b.dot(np.array([[theta0],[theta1]]))-y).sum()
djt1 = lambda X_b,y,theta0,theta1: 1/m*((X_b.dot(np.array([[theta0],[theta1]]))-y)*X_b[:,1].reshape(-1,1)).sum()
z = lambda theta0,theta1,theta0_touch,theta1_touch,X_b,y: djt0(X_b,y,theta0_touch,theta1_touch)*(theta0-theta0_touch)+djt1(X_b,y,theta0_touch,theta1_touch)*(theta1-theta1_touch)+j(X_b,y,theta0_touch,theta1_touch)