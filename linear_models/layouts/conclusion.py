import dash_html_components as html

conc = [
    html.H2("Conclusion", className="display-5"),
    html.P(
        [
            "That's it 💥! A comprehensive overview of how linear models work. I hope you enjoyed the examples and that the possibility to interactively adjust the parameters allowed you to take a look under the hood of the models. Don't hesitate to ",
            html.A("write me", href="mailto:philipp.stuerner@web.de"),
            " if you encounter any bugs or have any suggestions.",
        ]
    ),
    html.P(["Until next time,", html.Br(), "Philipp"]),
]

notes = [
    html.Hr(),
    html.P(
        [
            "Notes:",
            html.Br(),
            "Example, code snippets and math notations partly base on the fourth chapter of Aurélien Géron's excellent book ",
            html.A(
                "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition",
                href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/",
                target="_blank",
            ),
        ],
        style={"font-size": "0.7rem"},
    ),
]

layout = html.Div(
    [
        html.Div(conc, className="bs-component"),
        html.Div(notes, className="bs-component"),
    ]
)
