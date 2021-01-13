import numpy as np
import dash_bootstrap_components as dbc

external_scripts = [
    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML',
    '/static/mathjax-refresh.js',
    '/static/additional.js'
    ]
external_stylesheets = [
    'https://bootswatch.com/_assets/css/custom.min.css',
    '/static/additional.css',
    dbc.themes.LUX
    ]

def read_data():
    y_train_5 = np.load('data/y_train_5.npy')
    y_train = np.load('data/y_train.npy')
    y_scores = np.load('data/y_scores.npy')

    return y_train_5, y_train, y_scores