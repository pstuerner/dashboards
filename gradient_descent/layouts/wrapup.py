import dash_html_components as html
import dash_bootstrap_components as dbc

wrapup = dbc.Col([
        html.H2('Wrap-up', id='wrap_up'),
        html.P([
            """
            That's it! That was an intense post so kudos if you made it this far. I had actually planned to do a very short and crisp dashboard on Gradient Descent, but quickly realized that the algorithm and applications are too versatile and a more detailed elaboration is needed. I hope the examples and math notations helped you to get a better overview of Gradient Descent. The racetrack example is, in my opinion, a fun way to illustrate differences in the three algorithms and to show how different alternatives affect the final path. As always, feel free to email me if you have any questions, the examples are confusing, or you spot an error.
            """
        ]),
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center')