import dash_html_components as html

layout = html.Div(
    [
        html.P(
            """
    Welcome to Under the Hood 👋! A collection of interactive dashboards created to visualize and explain common machine learning algorithms.
    In the first edition of this series, we'll look at a widely used, relatively simple, yet very performant family of algorithms: linear models.
    """
        ),
        html.P(
            """
    Linear models were probably my first exposure to machine learning, though at that time I didn't really know what machine learning even is. I remember a lecture on econometrics in which we predicted a country's gross domestic product (GDP) using a linear regression. The professor couldn't really sell the topic in a sexy way, so I wasn't very enthusiastic about putting a straight line through a scatter plot and declared the lecture as utterly boring. Today, I believe that linear models are among the most beautiful flavors in machine learning. Compared to other machine learning algorithms, linear models are very simple. This has the advantage that you can always reconstruct how the final prediction was made. However, linear models should not be underestimated, as they are incredibly versatile and can solve a wide range of problems.
    """
        ),
        html.P(
            """
    In the following, we first deal with the simplest form of a linear model, namely an univariate linear regression. We then find out how to model nonlinear relationships by including polynomial features into the linear regression model. Furthermore, we see from the discrepancy between train and test errors that polynomial regressions and their degrees of freedom are a great extension, but should be used with caution. Finally, we find out that linear models are not only useful for solving regression but also classification problems. Let's get right to it.
    """
        ),
    ],
    className="bs-component",
    style={"margin-top": "0rem"},
)
