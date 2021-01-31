import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from sklearn import datasets

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


iris = datasets.load_iris()
X_iris = iris["data"][:, 3:].ravel()
y_iris = (iris["target"] == 2).astype(np.int).ravel()
df_iris = pd.DataFrame(np.c_[X_iris, y_iris], columns=["X", "y"])


def array2matrix(arr, max_rows=5, max_cols=5):
    rows = min(arr.shape[0], max_rows)
    cols = min(arr.shape[1], max_cols)
    if arr.shape[1] > cols:
        rows_idx = (
            [int(rows / 2)] if rows % 2 != 0 else [int(rows / 2), int(rows / 2 - 1)]
        )
    else:
        rows_idx = []

    bmatrix = "\\begin{pmatrix}"
    for row in range(0, rows):
        vdots = "&\\dots" if arr.shape[1] > cols and row in rows_idx else ""
        bmatrix += "&".join(arr[row][:cols].round(2).astype(str)) + vdots + "\\\\"

    if arr.shape[0] > rows:
        dots = [""] * cols
        if cols % 2 != 0:
            dots[int(cols / 2)] = "\\vdots"
        else:
            dots[int(cols / 2)] = "\\vdots"
            dots[int(cols / 2 - 1)] = "\\vdots"
    else:
        dots = [""]
    dots = "&".join(dots)
    bmatrix = bmatrix + dots + "\\end{pmatrix}"

    return bmatrix


hat_str = "\\hat"
theta_str = "\\theta"


def subsup(str_="", sub=None, sup=None):
    if sub != None:
        str_ += f"_{{{sub}}}"
    if sup != None:
        str_ += f"^{{{sup}}}"

    return str_


def theta(sub=None, sup=None):
    return subsup(theta_str, sub, sup)


def hat(str_=""):
    return f"{hat_str}{{{str_}}}"
