import dash_html_components as html

toc = html.Div([
    html.H3('Contents'),
    html.Ul([
        html.Li(html.A('What and how', href='#what_and_how')),
        html.Li(html.A('A simple example', href='#a_really_simple_example'))
    ])
])