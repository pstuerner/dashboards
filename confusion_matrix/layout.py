import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go

layout = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H1('Background'),
                            html.P("""
                            Classification tasks are among the most common problems in data science. Take a look at the following examples and I am sure you will find yourself in one, if not all of them:"""),
                            html.Ul([
                                html.Li('Netflix recommends a movie you actually like'),
                                html.Li('Spotify suggests a song that turns out to become one of your favorites'),
                                html.Li('Amazon advertises a book that fits your favorite genre'),
                                html.Li('Instagram promotes a channel based on your interests'),
                                html.Li('Your bank rejects your mortgage application'),
                                html.Li('Receiving an email from the son of the Nigerian prince who, for whatever reason, wants to share his fortune with you')
                            ]),
                            html.P("""
                            All of these examples have some sort of classification engine in the background that determines the most appropriate action. In my experience, the goal of a classification task in most cases is to make a recommendation. Books, songs, movies, products, channels, hotels - you name it. However, there are many more classification tasks, such as automatically classifying emails as spam or determining whether a customer is creditworthy or not."""),
                            html.Hr(className='my-4'),
                            html.H1('Confusion Matrix'),
                            html.P(
                                dcc.Markdown("""
                                One of the most important concepts in classification tasks is the confusion matrix, **which contains a set of metrics that help evaluate the performance of a classification algorithm**. The confusion matrix itself is quite simple, containing only the ground truth labels and the predictions of the classifier. Let us construct a simple classification task to understand what is contained in the confusion matrix.""")
                            ),
                            html.P(
                                dcc.Markdown("""
                                MNIST is a dataset that contains vectorized data of images of handwritten digits. It is a fairly popular dataset that has become the "Hello World!" for some machine learning problems. The most obvious classification task for this dataset is to classify each image of a handwritten digit to an integer.""")
                            ),
                            html.Div([
                                html.P([html.Img(src='/static/mnist_0.png', style={'width':'60px'}), ' → 0']),
                                html.P([html.Img(src='/static/mnist_5.png', style={'width':'60px'}), ' → 5']),
                                html.P([html.Img(src='/static/mnist_9.png', style={'width':'60px'}), ' → 9']),
                            ], style={'text-align':'center'}),
                            html.P(
                                dcc.Markdown("""
                                This is called multilabel classification because the algorithm tries to assign each image to multiple labels (digits 0 to 9). Since we want to keep it simple, we translate the usual MNIST classification task into a binary problem: classify whether an image is a five or not.""")
                            ),
                            html.Div([
                                html.P([html.Img(src="/static/mnist_0.png", style={'width':'60px'}), ' → False']),
                                html.P([html.Img(src='/static/mnist_5.png', style={'width':'60px'}), ' → True']),
                                html.P([html.Img(src='/static/mnist_9.png', style={'width':'60px'}), ' → False']),
                            ], style={'text-align':'center'}),
                            html.P(
                                dcc.Markdown("""
                                This makes the confusion matrix much more understandable, since we are only dealing with a 2x2 matrix. The confusion matrix works for any size of multilabel classification, it is just more visually appealing in the simple case.
                                """)
                            ),
                            html.P(
                                dcc.Markdown("""
                                For our binary classifier, the confusion matrix looks like the following:
                                """)
                            ),
                            html.Div([
                                html.Table([
                                    html.Tr([
                                        html.Td(colSpan=2, style={'border':'none'}),
                                        html.Td(html.H6('Classifier prediction'), colSpan=2, style={'vertical-align':'middle',  'border-top':'none', 'border-bottom':'none'}),
                                    ]),
                                    html.Tr([
                                        html.Td(colSpan=2, style={'border':'none'}),
                                        html.Td('-', style={'vertical-align':'middle', 'border-top':'none', 'border-right':'1px solid rgba(0,0,0,0.05)'}),
                                        html.Td('+', style={'vertical-align':'middle', 'border-top':'none'}),
                                    ]),
                                    html.Tr([
                                        html.Td(html.H6('Ground truth'),rowSpan=2, style={'vertical-align':'middle', 'border-top':'none', 'writing-mode':'tb-rl', 'transform':'rotate(-180deg)'}),
                                        html.Td('-', style={'vertical-align':'middle', 'border-top':'none'}),
                                        html.Td('True Negatives (TN)', style={'vertical-align':'middle', 'border-top':'none', 'border-right':'1px solid rgba(0,0,0,0.05)', 'background-color':'rgba(0, 100, 0, 0.3)'}),
                                        html.Td('False Positives (FP)', style={'vertical-align':'middle', 'border-top':'none', 'background-color':'rgba(139, 0, 0, 0.3)'}),
                                    ]),
                                    html.Tr([
                                        html.Td('+', style={'vertical-align':'middle'}),
                                        html.Td('False Negatives (FN)', style={'vertical-align':'middle', 'border-right':'1px solid rgba(0,0,0,0.05)', 'background-color':'rgba(139, 0, 0, 0.3)'}),
                                        html.Td('True Positives (TP)', style={'vertical-align':'middle', 'background-color':'rgba(0, 100, 0, 0.3)'}),
                                    ])
                                ], className='table table-hover', style={'text-align':'center'}),
                            ], className='col-lg-6 col-md-12 col-sm-12', style={'overflow':'auto', 'margin-left':'auto', 'margin-right':'auto'}),
                            html.Hr(className='my-4'),
                            html.H1('TN, FN, FP, TP'),
                            html.P(
                                dcc.Markdown("""
                                Once we have trained a classifier on existing training data, we can test its performance. To do this, we use the trained classifier to make predictions on unseen data and assign each result to the appropriate confusion matrix bin. There are four main bins that you should be aware of:
                                """)
                            ),
                            html.Ol([
                                html.Li(dcc.Markdown("True Negatives (TN): Observations that **do not belong** to the target label and were classified as **not belonging** to the target label. In our case: handwritten digits of 0, 3 and 9 that were **not classified** as 5s.")),
                                html.Li(dcc.Markdown("False Positives (FP): Observations that **do not belong** to the target label and were classified as **belonging** to the target label. In our case: handwritten digits of 0, 3 and 9 that were **classified** as 5s.")),
                                html.Li(dcc.Markdown("False Negatives (FN): Observations that **belong** to the target label and were classified as **not belonging** to the target label. In our case: handwritten digits of 5s that were **not classified** as 5s.")),
                                html.Li(dcc.Markdown("True Positives (TP): Observations that **belong** to the target label and were classified as **belonging** to the target label. In our case: handwritten digits of 5s that were **classified** as 5s.")),
                            ]),
                            html.P(
                                dcc.Markdown("""
                                From my experience, it's not a concept that gets stuck in your head right away. That's not because it's extremely complicated, but rather because it's a little difficult to distinguish between the terms. However, if you think about it often enough or put it into practice a few times then TN, FP, FN and TP will soon feel very natural! With every prediction result mapped to its confusion matrix bin, we have everything we need to compute some really handy metrics to help us evaluate the classifier's performance.
                                """)
                            ),
                            html.Hr(className='my-4'),
                            html.H1('Metrics'),
                            html.P(
                                dcc.Markdown("""
                                There are four metrics you should constantly keep in mind:
                                """)
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Accuracy", className="card-title"),
                                                html.P('$$ = \\frac{\\textrm{TN}+\\textrm{TP}}{\\textrm{TN}+\\textrm{FP}+\\textrm{FN}+\\textrm{TP}} $$')
                                            ]
                                        )
                                    )
                                , className='col-lg-3 col-md-6 col-sm-6'),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Precision", className="card-title"),
                                                html.P('$$ = \\frac{\\textrm{TP}}{\\textrm{TP}+\\textrm{FP}} $$')
                                            ]
                                        )
                                    )    
                                , className='col-lg-3 col-md-6 col-sm-6'),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Recall", className="card-title"),
                                                html.P('$$ = \\frac{\\textrm{TP}}{\\textrm{TP}+\\textrm{FN}} $$')
                                            ]
                                        )
                                    )
                                , className='col-lg-3 col-md-6 col-sm-6'),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("F1", className="card-title"),
                                                html.P('$$ = 2*\\frac{\\textrm{recall}*\\textrm{precision}}{\\textrm{recall}+\\textrm{precision}} $$')
                                            ]
                                        )
                                    )    
                                , className='col-lg-3 col-md-6 col-sm-6'),
                            ]),
                            html.Ol([
                                html.Li(dcc.Markdown("Accuracy: ratio of all correct predictions (TN, TP) to all predictions (TN, FP, FN, TP).")),
                                html.Li(dcc.Markdown("Precision: ratio of correctly classified positive labels (TP) to all positive classifications (TP, FP).")),
                                html.Li(dcc.Markdown("Recall: ratio of correctly classified positive labels (TP) to all actual positive labels (TP, FN).")),
                                html.Li(dcc.Markdown("F1: Weighted average of precision and recall.")),
                            ]),
                            html.P(
                                dcc.Markdown("""
                                Working with these metrics can be a bit tricky, as there are a few things to keep in mind in order to interpret the results correctly and draw the right conclusions.
                                Therefore, you should pay attention to the following "terms and conditions":
                                """)
                            ),
                            html.Ul([
                                html.Li(dcc.Markdown("""
                                Although accuracy is the most understandable metric, it is also the most dangerous of the four.
                                Remember that accuracy is a reliable metric only if the underlying data are not too skewed!
                                If the number of positive labels, in our case handwritten fives, is many times smaller than the number of non-positive labels, in our case all handwritten digits except fives, high accuracy is no guarantee of a good classifier.
                                This is because in the unbalanced case, high accuracy is associated with low recall.
                                """)),
                                html.Li(dcc.Markdown("""
                                For unbalanced data sets, it is useful to evaluate the classifier based on precision and recall while considering the tradeoff of the two metrics.
                                """)),
                                html.Li(dcc.Markdown("""
                                Precision and recall run in opposite directions.
                                High precision leads to low recall and vice versa.
                                Which metric is more important depends on the application.
                                Just consider whether false positives or false negatives are more critical.
                                For example, if you are classifying films for a children's channel, a false positive (a film that is not safe but is classified as safe) would be worse than a false negative (a film that is safe but is classified as not safe).
                                In this case, precision is the critical metric.
                                On the other hand, when classifying terrorist financing transactions, a false negative (a terrorist financing transaction that is classified as secure) would be worse than a false positive (a secure transaction that is classified as terrorist financing).
                                In this case, recall is the key metric.
                                """)),
                                html.Li(dcc.Markdown("""
                                If precision and recall are equally important, opt for the F1 score.
                                The F1 score is only high if both precision and recall have a high value and thus takes into account the tradeoff just described.
                                """)),
                            ]),
                            html.Hr(className='my-4'),
                            html.H1('Dashboard'),
                            html.P(
                                dcc.Markdown("""
                                This is all you need to know about the confusion matrix and its metrics. The following dashboard visualizes the distinction between TN, FP, FN and TP, the calculation of the metrics and the tradeoff between precision and recall.
                                """)
                            ),
                            html.P(
                                dcc.Markdown("""
                                The content of the dashboard is based on the binary classification task introduced earlier: predict whether an image of a handwritten number is a five. The predictions are made on a dataset of 60,000 images, which means that for each individual image a decision is made whether it is a five or not.
                                """)
                            ),
                            html.P(
                                dcc.Markdown("""
                                The decision is made by a Stochastic Gradient Descent (SGD) classifier previously trained on unseen data. Since classification algorithms are not the main focus of this dashboard, I will not go into detail. At this point it is only important to know that the final decision of the SGD classifier whether a handwritten number is a five or not depends on the value of a decision function and a threshold. If the value of the decision function is smaller than the threshold, the classifier predicts that it is not a five. If the value is larger than the threshold, the classifier predicts that it is a five.
                                """)
                            ),
                            html.P(
                                dcc.Markdown("""
                                The threshold of the algorithm can be changed and affects the overall performance of the classifier depending on the choice. The following three bullet points summarize the effects of changing the threshold:
                                """)
                            ),
                            html.Ul([
                                html.Li([
                                    'The larger the threshold, the more certain the algorithm must be to predict a five. The algorithm becomes ',
                                    html.A('more conservative', href='javascript:void(0);', id='more_conservative', style={'color':'blue','font-weight':'bold'}),
                                    ' which leads to many false negatives, few false positives, high precision and low recall.'
                                ]),
                                html.Li([
                                    'The smaller the threshold, the more uncertain the algorithm can be to predict a five. The algorithm becomes ',
                                    html.A('more aggressive', href='javascript:void(0);', id='more_aggressive', style={'color':'blue','font-weight':'bold'}),
                                    ' which leads to many false positives, few false negatives, high recall and low precision.'
                                ]),
                                html.Li([
                                    'The sweet spot, a ',
                                    html.A('high F1 score', href='javascript:void(0);', id='sweet_spot', style={'color':'blue','font-weight':'bold'}),
                                    ' is somewhere in the middle.'
                                ])
                            ]),
                            html.P(
                                dcc.Markdown("""
                                That's all you need! Click on the hyperlinks in the bullet points above to set a threshold for that scenario. Play around with the threshold slider until the changes to the confusion matrix, its metrics, and the graphs make sense. Check the example data section to see which handwritten digits are classified as TN, FP, FN or TP.
                                """)
                            ),
                        ], className='jumbotron', style={'padding':'2rem 2rem'})
                    ], className='bs-component')
                ], className='col-lg-12')
            ], className='row')
        ], className='bs-docs-section', style={'margin-top':'0em'}),

        html.Div([
            html.Div(id='loader_wrapper', children=[
                dcc.Loading(color='transparent', children=[
                    html.Div([
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Threshold", className="card-title", style={'text-align':'center'}),
                                        dcc.Slider(
                                            id='threshold_slider',
                                            min=-50000,
                                            max=50000,
                                            step=5000,
                                            value=0,
                                            marks={-50000: {'label': '-50000', 'style': {'color': '#fff'}}, -25000: {'label': '-25000', 'style': {'color': '#fff'}}, 0: {'label': '0', 'style': {'color': '#fff'}}, 25000: {'label': '25000', 'style': {'color': '#fff'}}, 50000: {'label': '50000', 'style': {'color': '#fff'}}},
                                            updatemode='mouseup',
                                            disabled=False
                                        ),
                                        html.H5(id='threshold_output', className="card-title", style={'text-align':'center'}),
                                    ], style={'padding':'0.25rem'}
                                ), className='card text-white bg-dark mb-3', style={'height':'105px', 'background-color':'#343a40ed !important'})
                        ),
                    ], className='sticky col-lg-4 col-md-6 col-sm-6', style={'margin-left':'auto', 'margin-right':'auto', 'height':'105px'}),            
                        
                    html.Div([
                        html.Div([
                            dbc.Row([
                                html.Div([
                                    html.Div([
                                        html.H3('Confusion matrix'),
                                        dbc.FormGroup(
                                            [
                                                dbc.RadioItems(
                                                    options=[
                                                        {"label": "Absolute", "value": 'abs'},
                                                        {"label": "Relative", "value": 'rel'},
                                                    ],
                                                    value='abs',
                                                    id="absrel_input",
                                                    inline=True
                                                ),
                                            ]
                                        ),
                                    ], className='col-12', style={'height':'60px', 'margin-bottom':'15px'}),
                                    html.Div([
                                        html.Table([
                                            html.Tr([
                                                html.Td(id='tn', children=[0], style={'background-color':'rgba(0, 100, 0, 0.3)', 'border-top':'none', 'border-right':'1px solid rgba(0,0,0,0.05)'}),
                                                html.Td(id='fp', children=[0], style={'background-color':'rgba(139, 0, 0, 0.3)', 'border-top':'none'}),
                                            ]),
                                            html.Tr([
                                                html.Td(id='fn', children=[0], style={'background-color':'rgba(139, 0, 0, 0.3)', 'border-right':'1px solid rgba(0,0,0,0.05)'}),
                                                html.Td(id='tp', children=[0], style={'background-color':'rgba(0, 100, 0, 0.3)'}),
                                            ])
                                        ], className='table table-hover', style={'text-align':'center', 'line-height':'40px'})
                                    ]),
                                ], className='col-lg-6 col-md-6 col-sm-12'),
                                html.Div([
                                    html.Div([
                                        html.H3('Example data')
                                    ], className='col-12', style={'height':'60px', 'margin-bottom':'15px'}),
                                    html.Div([
                                        html.Table([
                                            html.Tr([
                                                html.Td(id='tn_digits', children=[0], style={'background-color':'rgba(0, 100, 0, 0.3)', 'border-top':'none', 'border-right':'1px solid rgba(0,0,0,0.05)', 'max-width': 0, 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap'}),
                                                html.Td(id='fp_digits', children=[0], style={'background-color':'rgba(139, 0, 0, 0.3)', 'max-width': 0, 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap'}),
                                            ]),
                                            html.Tr([
                                                html.Td(id='fn_digits', children=[0], style={'background-color':'rgba(139, 0, 0, 0.3)', 'border-right':'1px solid rgba(0,0,0,0.05)', 'max-width': 0, 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap'}),
                                                html.Td(id='tp_digits', children=[0], style={'background-color':'rgba(0, 100, 0, 0.3)', 'max-width': 0, 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap'}),
                                            ])
                                        ], className='table table-hover', style={'width':'100%', 'line-height':'40px'})
                                    ]),
                                ], className='col-lg-6 col-md-6 col-sm-12'),
                            ]),
                        ], className='bs-component'),
                    ], className='bs-docs-section'),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.H3('Metrics')
                                ], className='col-lg-12 col-md-12 col-sm-12')
                            ], className='row'),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card([
                                        html.Div([
                                            'Accuracy'
                                        ], className='card-header'),
                                        dbc.CardBody(
                                            [
                                                html.P(id='accuracyeq')
                                        ], className='card border-primary', style={'margin-bottom':0})
                                    ], style={'height':'175px'}),  className='col-lg-3 col-md-6 col-sm-6'
                                ),
                                dbc.Col(
                                    dbc.Card([
                                        html.Div([
                                            'Precision'
                                        ], className='card-header'),
                                        dbc.CardBody(
                                            [
                                                html.P(id='precisioneq')
                                        ], className='card border-primary', style={'margin-bottom':0})
                                    ], style={'height':'175px'}),  className='col-lg-3 col-md-6 col-sm-6'
                                ),
                                dbc.Col(
                                    dbc.Card([
                                        html.Div([
                                            'Recall'
                                        ], className='card-header'),
                                        dbc.CardBody(
                                            [
                                                html.P(id='recalleq')
                                        ], className='card border-primary', style={'margin-bottom':0})
                                    ], style={'height':'175px'}),  className='col-lg-3 col-md-6 col-sm-6'
                                ),
                                dbc.Col(
                                    dbc.Card([
                                        html.Div([
                                            'F1'
                                        ], className='card-header'),
                                        dbc.CardBody(
                                            [
                                                html.P(id='f1eq')
                                        ], className='card border-primary', style={'margin-bottom':0})
                                    ], style={'height':'175px'}),  className='col-lg-3 col-md-6 col-sm-6')
                            ]),
                        ], className='bs-component'),
                    ], className='bs-docs-section'),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.H3('Graphs')
                                ], className='col-lg-12 col-md-12 col-sm-12')
                            ], className='row'),
                            html.Div([
                                html.Div([
                                    dcc.Graph(
                                            id='precision_recall_graph',
                                            figure=go.Figure(),
                                            style={'width':'100%','height':'100%'},
                                            config={'staticPlot':True}
                                        )
                                ], className='col-lg-12'),
                            ], className='row'),
                            html.Div([
                                html.Div([
                                    dcc.Graph(
                                            id='auc_graph',
                                            figure=go.Figure(),
                                            style={'width':'100%','height':'100%'},
                                            config={'staticPlot':True}
                                        )
                                ], className='col-lg-6'),
                                html.Div([
                                    dcc.Graph(
                                            id='aucpr_graph',
                                            figure=go.Figure(),
                                            style={'width':'100%','height':'100%'},
                                            config={'staticPlot':True}
                                        )
                                ], className='col-lg-6'),
                            ], className='row')
                        ], className='bs-component'),
                    ], className='bs-docs-section'),
                ]),
            ]),
        ], className='bs-docs-section'),
    ], className='container', style={'padding-bottom':'150px', 'max-width':'100%'})
