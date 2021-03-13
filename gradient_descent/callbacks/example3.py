import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_html_components as html

from sklearn.metrics import mean_squared_error
from dash.dependencies import Input, Output, State
from util import j, djt1
from app import app


@app.callback(Output('example3_div_mse','children'),[Input('example3_div_theta1init','children'),Input('example3_div_theta1hist','children'), State('data','children'), State('best_theta','children')])
def example3_div_mse(theta1_init, theta1_hist, data, best_theta):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context
    
    if ctx.triggered[0]['prop_id'] == 'theta1_init.children':
        theta1 = theta1_init
    else:
        theta1 = theta1_hist[-1]
    
    mse = j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)
        
    return [html.H3('MSE'),f'{mse:.2f}']
    
@app.callback(Output('example3_div_theta1init','children'),[Input('example3_button_reset','n_clicks')])
def example3_theta1init(n_clicks):
    ctx = dash.callback_context
    
    if 'reset' in ctx.triggered[0]['prop_id']:
        v = int(theta1_best)
        while theta1_best+1.5>v>theta1_best-1.5:
            v = round(random.uniform(-1.5,7.5),2)
    else:
        v = 6
                
    return v

@app.callback(
    Output('example3_div_theta1hist','children'),
    [Input('example3_button_nextstep','n_clicks'),
     Input('example3_div_theta1init','children'),
     State('example3_div_theta1hist','children'),
     State('example3_input_eta','value'),
     State('best_theta','children'),
     State('data','children')]
)
def example3_theta1hist(n_clicks,theta1_init,theta1_hist,eta,theta_best,data):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'theta1init' in ctx:
        return [theta1_init]
    else:
        gradient = djt1(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta_best[0],theta1_hist[-1])
        theta1_new = theta1_hist[-1] - eta * gradient
        return theta1_hist+[theta1_new]

@app.callback(
    Output('example3_graph_regression', 'figure'),
    [Input('example3_div_theta1init','children'),
     Input('example3_div_theta1hist','children'),
     State('example3_graph_regression','figure'),
     State('data','children'),
     State('best_theta','children')]
)
def example3_regression(theta1_init, theta1_hist, fig, data, best_theta):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'theta1init' in ctx:
        return go.Figure(
                data=[
                    go.Scatter(x=df['X'], y=df['y'], mode='markers', name='Data', hovertemplate='(%{x:.2f}, %{y:.2f})'),
                    go.Scatter(x=df['X'], y=best_theta[0]+theta1_init*df['X'], name='Initialization', line = dict(color='red', width=2, dash='dot'), hovertemplate='(%{x:.2f}, %{y:.2f})')              
                ],
                layout = go.Layout(
                    yaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='y',
                        range = [-1,15]
                    ),
                    xaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='X',
                        range = [0,2]
                    ),
                    margin=dict(t=5,b=5,l=5,r=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(176,196,222,0.9)",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                    ),
                )
        )
    elif 'theta1hist' in ctx:
        showlegend = True if len(theta1_hist)==2 else False
        
        figure = go.Figure(data=fig['data'], layout=fig['layout'])
        figure.add_trace(go.Scatter(x=df['X'], y=best_theta[0]+theta1_hist[-1]*df['X'], mode='lines', line=dict(color='black'), name=f'Steps', showlegend=showlegend, hovertemplate='(%{x:.2f}, %{y:.2f})'))
        return figure
    
@app.callback(
    Output('example3_graph_lossfunction', 'figure'),
    [Input('example3_div_theta1hist','children'),
     Input('example3_div_theta1init','children'),
     State('example3_graph_lossfunction','figure'),
     State('data','children'),
     State('best_theta','children')]
)
def example3_lossfunction(theta1_hist, theta1_init, fig, data, best_theta):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'theta1init' in ctx:
        theta1 = theta1_init
        
        figure = go.Figure(
                data=[
                    go.Scatter(x=np.arange(best_theta[1]-5,best_theta[1]+5,0.1), y=[j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1) for theta1 in np.arange(best_theta[1]-5,best_theta[1]+5,0.1)], name='Cost function', hovertemplate='(%{x:.2f}, %{y:.2f})'),
                    go.Scatter(x=[theta1_init], y=[j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1_init)], mode='markers', name='Initialization', marker=dict(color='red', size=10), hovertemplate='(%{x:.2f}, %{y:.2f})')
                ],
                layout = go.Layout(
                    yaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='MSE Cost',
                        range=[-0.5,j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],best_theta[1]+5)],
                        autorange=False
                    ),
                    xaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='theta_1',
                        range = [best_theta[1]-5,best_theta[1]+5],
                        autorange=False
                    ),
                    margin=dict(t=5,b=5,l=5,r=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(176,196,222,0.9)",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                    ),
                )
        )
    elif 'theta1hist' in ctx:
        showlegend = True if len(theta1_hist)==2 else False
        theta1_prev, theta1 = theta1_hist[-2:]
        
        figure = go.Figure(data=fig['data'], layout=fig['layout'])
        figure.add_trace(go.Scatter(x=[theta1], y=[j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)], mode='markers', name=f'Steps', showlegend=showlegend, marker=dict(color='black', size=10), hovertemplate='(%{x:.2f}, %{y:.2f})'))
        
        x1,y1 = theta1_prev,j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1_prev)
        x0,y0 = theta1,j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)
        
        figure.add_shape(dict(
                    type="path",
                    path=f"M {x1},{y1} Q {(x1+x0)/2},{max(y1,y0)+abs(y1-y0)} {x0},{y0}",
                    line_color="black",
                ))
    else:
        figure = fig
    
    idx = [i for i,x in enumerate(figure['data']) if x.name=='Gradient']
    if idx != []:
        fdata = list(figure['data'])
        fdata.pop(idx[0])
        figure['data'] = tuple(fdata)
    
    m = djt1(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)
    y = j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)
    x = theta1
    b = y-x*m
    f = lambda x: b+m*x
    figure.add_trace(
        go.Scatter(x=[x-1,x+1], y=[f(x-1),f(x+1)], mode='lines', line = dict(color='red', width=2), name='Gradient', hovertemplate='(%{x:.2f}, %{y:.2f})')
    )
    
    return figure