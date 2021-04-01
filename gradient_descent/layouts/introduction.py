import dash_html_components as html
import dash_bootstrap_components as dbc

introduction = dbc.Col([
        html.P("""
        Some things in life are important such as giving your parents a call from time to time, brushing your teeth, drinking enough water or making sure not to forget your fiancé's birthday. The list of important things is of course much longer and depends on the individual, but I'm pretty sure that Gradient Descent should be on everyone's list. Why? Because the world is full of problems that revolve around finding an optimal solution and Gradient Descent, which is a generic optimization algorithm, is capable of finding these solutions! It is such an important algorithm that many people assume that it must be really complicated. Spoiler warning: it's not 🙏.
        """),
        html.P("""
        I was really looking forward to this set of dashboards since Gradient Descent is just perfect to visualize and so easy to understand as soon as you actually see what's going on. The following sections will lead you through the entire algorithm, show you what's happening under the hood and what the underlying math looks like. I'll start with an extremely simple example to set the scene and let you know what Gradient Descent even is, what it's used for and what the general idea behind the algorithm is. Afterwards, we'll extend the first example to find out how Gradient Descent works when the underlying problem becomes more complex. Finally, we'll also explore what Gradient Descent's pitfalls are and learn more about further variations to tackle these obstacles. In the final section we are going to recap on all different Gradient Descent variations we've seen so far and compare them in terms of prediction error and computation time. Sounds alright 😎? Go ahead, check the table of contents and start exploring the first section.
        """)
    ], xs=12, sm=12, md=12, lg=8, className='bs-component center')