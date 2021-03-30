import numpy as np
import dash_bootstrap_components as dbc

from sklearn.linear_model import LinearRegression

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

X = np.load('X.npy') #= 2 * np.random.rand(m, 1)
y = np.load('y.npy') #= 4 + 3 * X + np.random.randn(m, 1)
m = len(X)

lr = LinearRegression()
lr.fit(X,y)
theta0_best, theta1_best = lr.intercept_[0], lr.coef_[0][0]

j = lambda X_b,y,theta0,theta1: 1/2*1/m*((X_b.dot(np.array([[theta0],[theta1]]))-y)**2).sum()
djt0 = lambda X_b,y,theta0,theta1: 1/m*(X_b.dot(np.array([[theta0],[theta1]]))-y).sum()
djt1 = lambda X_b,y,theta0,theta1: 1/m*((X_b.dot(np.array([[theta0],[theta1]]))-y)*X_b[:,1].reshape(-1,1)).sum()
z = lambda theta0,theta1,theta0_touch,theta1_touch,X_b,y: djt0(X_b,y,theta0_touch,theta1_touch)*(theta0-theta0_touch)+djt1(X_b,y,theta0_touch,theta1_touch)*(theta1-theta1_touch)+j(X_b,y,theta0_touch,theta1_touch)

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