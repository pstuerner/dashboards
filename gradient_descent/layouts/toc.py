import dash_html_components as html
import dash_bootstrap_components as dbc

toc = dbc.Col([
    html.H1('Contents'),
    html.Ul([
        html.Li(html.A('(Batch) Gradient Descent', href='#batch_gradient_descent')),
        html.Li(html.A('A Really Simple Example', href='#a_really_simple_example')),
        html.Li(html.A('A Less Simple Example', href='#a_less_simple_example')),
        html.Li([
            html.A('Beyond Batch Gradient Descent', href='#beyond_batch_gradient_descent'),
            html.Ul([
                html.Li(html.A('Stochastic Gradient Descent', href='#stochastic_gradient_descent')),
                html.Li(html.A('Mini-batch Gradient Descent', href='#minibatch_gradient_descent')),
            ])
        ]),
        html.Li(html.A('Gradient Descent Racetrack', href='#gradient_descent_racetrack')),
    ])
], xs=12, sm=12, md=12, lg=8, className='bs-component center')