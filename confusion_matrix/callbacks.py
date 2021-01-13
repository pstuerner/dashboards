import dash
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from util import read_data
from app import app

y_train_5, y_train, y_scores = read_data()
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
fpr, tpr, roc_thresholds = roc_curve(y_train_5, y_scores)  

@app.callback(
    [Output('accuracyeq','children'),
    Output('precisioneq','children'),
    Output('recalleq','children'),
    Output('f1eq','children'),],
    [Input('threshold_slider', 'value'),]
)
def update_scores(value):
    y_pred_5 = (y_scores >= value)
    cm = confusion_matrix(y_train_5, y_pred_5)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(recall*precision)/(recall+precision)

    accuracy_eq = f'$$ = \\frac{{{tp}+{tn}}}{{{tp}+{fp}+{tn}+{fn}}}$$ $$ = {accuracy*100:.1f}\% $$'
    precision_eq = f'$$ = \\frac{{{tp}}}{{{tp}+{fp}}}$$ $$ = {precision*100:.1f}\% $$'
    recall_eq = f'$$ = \\frac{{{tp}}}{{{tp}+{fn}}}$$ $$ = {recall*100:.1f}\% $$'
    f1_eq = f'$$ = 2*\\frac{{{recall:.2f}*{precision:.2f}}}{{{recall:.2f}+{precision:.2f}}}$$ $$ = {f1*100:.1f}\% $$'
    return accuracy_eq, precision_eq, recall_eq, f1_eq

@app.callback(
    [Output('tn', 'children'),
    Output('fp', 'children'),
    Output('fn', 'children'),
    Output('tp', 'children'),
    Output('threshold_output','children'),],
    [Input('threshold_slider', 'value'),
    Input('absrel_input', 'value')])
def update_cm(value, absrel):
    normalize = None if absrel=='abs' else 'true'
    y_pred_5 = (y_scores >= value)
    cm = confusion_matrix(y_train_5, y_pred_5,normalize=normalize)

    if absrel=='abs':
        return [
            f'TN:   {cm[0][0]:,}',
            f'FP:   {cm[0][1]:,}',
            f'FN:   {cm[1][0]:,}',
            f'TP:   {cm[1][1]:,}',
            value
        ]
    else:
        return [
            f'TN:   {cm[0][0]*100:.1f}%',
            f'FP:   {cm[0][1]*100:.1f}%',
            f'FN:   {cm[1][0]*100:.1f}%',
            f'TP:   {cm[1][1]*100:.1f}%',
            value
        ]

@app.callback(
    [Output('tn_digits', 'children'),
    Output('fp_digits', 'children'),
    Output('fn_digits', 'children'),
    Output('tp_digits', 'children'),],
    [Input('threshold_slider', 'value')])
def update_cm_digits(value):
    y_pred_5 = (y_scores >= value)
    
    tns = y_train[(y_train!=5) & (y_pred_5==False)]
    fps = y_train[(y_train!=5) & (y_pred_5==True)]
    fns = y_train[(y_train==5) & (y_pred_5==False)]
    tps = y_train[(y_train==5) & (y_pred_5==True)]

    tns_rand = np.random.choice(tns, size=min(10,len(tns)), replace=False)
    fps_rand = np.random.choice(fps, size=min(10,len(fps)), replace=False)
    fns_rand = np.random.choice(fns, size=min(10,len(fns)), replace=False)
    tps_rand = np.random.choice(tps, size=min(10,len(tps)), replace=False)

    r = []
    for x in [tns_rand, fps_rand, fns_rand, tps_rand]:
        if len(x)<10:
            r.append(f"{', '.join(x.astype(str))}")
        else:
            r.append(f"{', '.join(x.astype(str))}, ...")

    return r

@app.callback(
    [Output('precision_recall_graph', 'figure'),
    Output('auc_graph', 'figure'),
    Output('aucpr_graph', 'figure')],
    [Input('threshold_slider', 'value')])
def update_graphs(value):
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], name='Recall'))
    pr_fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name='Precision'))
    pr_fig.add_trace(go.Scatter(showlegend=False, x=[-50000,value], y=[precisions[np.argmax(thresholds>=value)],precisions[np.argmax(thresholds>=value)]], line=dict(color='red', width=1, dash='dash')))
    pr_fig.add_trace(go.Scatter(showlegend=False, x=[value,value], y=[0,precisions[np.argmax(thresholds>=value)]], line=dict(color='red', width=1, dash='dash')))
    pr_fig.add_trace(go.Scatter(showlegend=False, x=[-50000,value], y=[recalls[np.argmax(thresholds>=value)],recalls[np.argmax(thresholds>=value)]], line=dict(color='red', width=1, dash='dash')))
    pr_fig.add_trace(go.Scatter(showlegend=False, x=[value,value], y=[0,recalls[np.argmax(thresholds>=value)]], line=dict(color='red', width=1, dash='dash')))
    pr_fig.update_layout(
        height=250,
        xaxis={'range':[-50000,50000]},
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99
        )
    )

    lfpr = fpr[np.argmax(roc_thresholds<=value)]
    ltpr = tpr[np.argmax(roc_thresholds<=value)]
    auc_fig = go.Figure()
    auc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC'))
    auc_fig.add_trace(go.Scatter(showlegend=False, x=[0,lfpr], y=[ltpr,ltpr], line=dict(color='red', width=1, dash='dash')))
    auc_fig.add_trace(go.Scatter(showlegend=False, x=[lfpr,lfpr], y=[0,ltpr], line=dict(color='red', width=1, dash='dash')))
    auc_fig.update_layout(
        height=250,
        xaxis={'range':[0,1]},
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99
        )
    )

    lp = precisions[np.argmax(thresholds>=value)]
    lr = recalls[np.argmax(thresholds>=value)]
    aucpr_fig = go.Figure()
    aucpr_fig.add_trace(go.Scatter(x=precisions, y=recalls, name='Precision-Recall'))
    aucpr_fig.add_trace(go.Scatter(showlegend=False, x=[0,lp], y=[lr,lr], line=dict(color='red', width=1, dash='dash')))
    aucpr_fig.add_trace(go.Scatter(showlegend=False, x=[lp,lp], y=[0,lr], line=dict(color='red', width=1, dash='dash')))
    aucpr_fig.update_layout(
        height=250,
        xaxis={'range':[0,1]},
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99
        )
    )

    return pr_fig, auc_fig, aucpr_fig

@app.callback(
    Output('threshold_slider','value'),
    [Input('more_conservative','n_clicks'),
    Input('more_aggressive','n_clicks'),
    Input('sweet_spot','n_clicks')]
)
def hyperlink_update(mc, ma, spot):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'more_conservative' in changed_id:
        return 40000
    elif 'more_aggressive' in changed_id:
        return -40000
    elif 'sweet_spot' in changed_id:
        return -2000
    else:
        return 0