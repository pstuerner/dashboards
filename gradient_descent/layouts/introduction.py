import dash_html_components as html
import dash_bootstrap_components as dbc

introduction = html.Div([
    dbc.Row([
        html.P("""
        Some things in life are important such as giving your parents a call from time to time, brushing your teeth, drinking enough water or making sure not to forget your fiancé's birthday. The list of important things is of course much longer and depends on the individual, but I'm pretty sure that gradient descent should be on everyone's list. Why? Because the world is full of problems that revolve around finding an optimal solution and gradient descent, which is a generic optimization algorithm, is capable of finding these solutions! It is such an important algorithm that many people assume that it must be really complicated. Spoiler warning: it's not.
        """),
        html.P("""
        I was really looking forward to this dashboard since gradient descent is just perfect for fancy visualizations and so easy to understand as soon as you actually see what's going on. The following sections will lead you through the entire algorithm, show you what's happening under the hood and what the underlying math looks like. I'll start with an extremely simple example to set the scene and let you know what gradient descent even is, what it's used for and what the general idea behind the algorithm is. Afterwards, we'll extend the first example to find out how gradient descent works when the underlying problem becomes more complex. Finally, we'll also explore what gradient descent's pitfalls are and learn more about further variations to tackle these obstacles. Sounds alright? Go ahead, check the table of contents and start exploring the first section.
        """)
    ])
], className="bs-component")