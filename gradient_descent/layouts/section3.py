import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

section3_layout = html.Div([
    dbc.Col([
        html.H2('Beyond Batch Gradient Descent', id='beyond_batch_gradient_descent'),
        html.P("""
        So what is wrong with Batch Gradient Descent? You have surely noticed that each additional model parameter results in another partial derivative of the cost function. For a model with three parameters, for example, the following situation would emerge: $$ \\textrm{MSE}(\\theta) = \\frac{{1}}{{2m}}\\sum_{{i=1}}^{{m}}{{(\\theta^Tx^i-y^i)^2}}, \\: \\textrm{with} \\: \\theta=\\begin{pmatrix}\\theta_0\\\\\\theta_1\\\\\\theta_2\\end{pmatrix} $$ $$ \\textrm{MSE}_{{\\theta_0}}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{0}}^{{i}} $$ $$ \\textrm{MSE}_{{\\theta_1}}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{1}}^{{i}} $$ $$ \\textrm{MSE}_{{\\theta_2}}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{2}}^{{i}} $$
        """),
        html.P([
            "Cumbersome! However, this is not a problem, because it is not necessary to determine the gradient by individual evaluation of the partial derivatives. It is far more convenient to determine a so-called gradient vector $\\nabla_{{\\theta}}\\textrm{MSE}(\\theta)$ to determine the entire gradient in one go. The determination of the gradient vector is more complicated and outside the scope of this post. In case you are curious about it, I recommend this answer on ",
            html.A('StackOverflow', target='_blank', href='https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables'),
            " and these ",
            html.A('lecture notes', target='_blank', href='http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec10_LeastSquaresRegression.pdf'),
            ". The gradient vector $\\nabla_{{\\theta}}\\textrm{MSE}(\\theta)$ looks like this: $$ \\nabla_{\\theta}\\textrm{MSE}(\\theta)=\\begin{pmatrix}\\frac{\\partial}{\\partial{\\theta_0}}\\textrm{MSE}(\\theta)\\\\\\frac{\\partial}{\\partial{\\theta_1}}\\textrm{MSE}(\\theta)\\\\\\vdots\\\\\\frac{\\partial}{\\partial{\\theta_n}}\\textrm{MSE}(\\theta)\\end{pmatrix}=\\frac{1}{m}X^T(X\\theta-y) $$"
        ]),
        html.P("""
        This is a neat trick and can save you some work. The following minimal example shows that the gradient vector leads to the same result as the more time-consuming way of evaluating the partial derivatives one by one.
        """),
        dcc.Markdown(
            [
            f"""
            ```python
            >>> import numpy as np
            >>> 
            >>> m = 50
            >>> X = 2 * np.random.rand(m, 2)
            >>> X_b = np.c_[np.ones(len(X)), X]
            >>> y = 4 + 3 * X[:,0].reshape(-1,1) + np.random.randn(m, 1)
            >>> 
            >>> theta0, theta1, theta2 = 2, 2, 2
            >>> theta = np.array([[theta0], [theta1], [theta2]])
            >>> 
            >>> dMSEdt0 = lambda X,y,theta: 1/m*((X.dot(theta)-y)*X[:,0].reshape(-1,1)).sum()
            >>> dMSEdt1 = lambda X,y,theta: 1/m*((X.dot(theta)-y)*X[:,1].reshape(-1,1)).sum()
            >>> dMSEdt2 = lambda X,y,theta: 1/m*((X.dot(theta)-y)*X[:,2].reshape(-1,1)).sum()
            >>> 
            >>> nabla = lambda X,y,theta: 1/m*X.T.dot(X.dot(theta)-y)
            >>> 
            >>> gradients_manual = np.array([[dMSEdt0(X_b,y,theta)],[dMSEdt1(X_b,y,theta)],[dMSEdt2(X_b,y,theta)]])
            >>> gradients_nabla = nabla(X_b,y,theta)
            >>> 
            >>> gradients_manual
            array([
                [-0.90448282],
                [-1.58994555],
                [-0.11876997]
            ])
            >>> gradients_nabla
            array([
                [-0.90448282],
                [-1.58994555],
                [-0.11876997]
            ])
            ```
            """
            ]
        ),
        html.P(dcc.Markdown("""
        Although the gradient vector is an enormous improvement, it does not solve the root of the problem: $X$. The reason is that each gradient descent step requires evaluating the **entire** training dataset $X$. This means that batch gradient descent becomes terribly slow on very large datasets. Interestingly, however, Gradient Descent scales very well when the number of features explodes (e.g. several hundred thousand) and performs significantly better than, for example, the Normal Equation. There are two main variations, namely Stochastic Gradient Descent and Mini-batch Gradient Descent, which prevent the need to use the entire training dataset at each step to calculate the gradient.
        """)),
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        html.H3('Stochastic Gradient Descent', id='stochastic_gradient_descent'),
        html.P("""
        Unlike Batch Gradient Descent, Stochastic Gradient Descent uses a random instance in the training data set to calculate the gradient for the next step. So you could say that Stochastic Gradient Descent is the extreme counterexample to Batch Gradient Descent. The former uses the minimum number of instances, while the latter uses the largest possible number of instances. The advantage is obvious: Stochastic Gradient Descent is fast. Regardless of the size of the data set.
        """),
        html.P("""
        Furthermore, the risk of getting stuck in a local optimum is lower. Not all cost functions are convex and perfectly bowl-shaped. Depending on the problem, a cost function may consist of multiple local minima and plateaus, making the approximation either more difficult or the result misleading. Stochastic Gradient Decent has the advantage that because of the built-in randomness, the algorithm does not take a direct path to the optimum, but rather jumps back and forth on the cost function. Thus, there is the possibility that Stochastic Gradient Descent moves away from a local optimum to find the global one.
        """),
        html.P("""
        The disadvantage is that the algorithm will arrive somewhere near the optimum but won't settle there for certain. Since one random instance is used to determine the gradient per step, Stochastic Gradient Descent jumps around the best solution even near the optimum. This means that the final parameters are good, but not optimal. One way to improve the convergence to the optimal solution is to gradually reduce the step size of Gradient Descent. This approach is referred to as the learning schedule and simply means that the algorithm takes large steps as long as it is far and small steps as soon as it is close to the optimum.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
    dbc.Col([
        html.H3('Mini-batch Gradient Descent', id='minibatch_gradient_descent'),
        html.P("""
        Now that you know Batch and Stochastic Gradient Descent, Mini-batch Gradient Descent is not difficult to explain. Basically, it is a middle way of the two Gradient Descent variants already explained. Again, the number of training instances used to determine the gradient is important. Mini-batch Gradient Descent works with so-called mini-batches and therefore uses more instances than Stochastic but less than Batch Gradient Descent.
        """),
        html.P("""
        Similar to Stochastic Gradient Descent, the used instances are randomly selected. Unsurprisingly, the path of Mini-batch Gradient Descent is more similar to the jumpy character of Stochastic Gradient Descent than to the smooth path of Batch Gradient Descent. The biggest advantage over Stochastic Gradient Descent is that the performance of the algorithm can be improved by hardware optimization (e.g. faster matrix operations by using GPUs).
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center'), 
])