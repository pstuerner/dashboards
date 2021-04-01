import dash
import os
import sympy as sp
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import random
import datetime

from functools import wraps
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from scipy import stats
from data import DATA_PATH
from app import app


def steps_check(steps, min, max):
    if steps is None or not min<=steps<=max:
        return False
    else:
        return True

def recordTime(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start_time = datetime.datetime.now()
        res=func(*args,**kwargs)
        end_time = datetime.datetime.now()
        t = end_time-start_time
        return res+(t.microseconds,)
    return wrapper

@recordTime
def bgd(X_b, y, theta0, theta1, eta, epoch, learning_schedule):
    theta = np.array([[theta0],[theta1]])
    eta = learning_schedule(eta, epoch)
    gradients = 2/len(X_b) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    
    return theta[0][0], theta[1][0], eta, epoch+1, epoch+1

@recordTime
def sgd(X_b, y, theta0, theta1, eta, epoch, batch, learning_schedule):
    if (batch+1)%10==0:
        epoch += 1
        batch = 0
    else:
        batch += 1
    
    theta = np.array([[theta0],[theta1]])
    eta = learning_schedule(eta, epoch)
    random_index = np.random.randint(len(X_b))
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    theta = theta - eta * gradients
    
    return theta[0][0], theta[1][0], eta, epoch, batch

@recordTime
def mbgd(X_b, y, theta0, theta1, eta, epoch, batch, batch_size, learning_schedule):
    if (batch+1)%10==0: #(batch+1)%(m//batch_size)==0
        epoch += 1
        batch = 0
    else:
        batch +=1
    
    theta = np.array([[theta0],[theta1]])
    eta = learning_schedule(eta, epoch)
    random_indices = np.random.choice(len(X_b), size=batch_size, replace=True)
    xi = X_b[random_indices]
    yi = y[random_indices]
    gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)
    theta = theta - eta * gradients
    
    return theta[0][0], theta[1][0], eta, epoch, batch

theta_init = np.array([[5],[-3]])
eta_init = 0.1
epoch_init = 0
batch_init = 0
epochs = 100
initial_learning_rate = 0.1
drop_rate = 0.5
drop_epochs = 10
decay = initial_learning_rate / epochs

constant_learning_schedule = lambda lr,_: lr
time_learning_schedule = lambda lr,epoch: lr * 1 / (1 + decay * epoch)
step_learning_schedule = lambda _,epoch: initial_learning_rate*drop_rate**math.floor(epoch/drop_epochs)

lr_dict = {
    'constant': constant_learning_schedule,
    'time': time_learning_schedule,
    'step': step_learning_schedule
}

ranks_dict = {
    1:'1 🥇',
    2:'2 🥈',
    3:'3 🥉'
}


@app.callback(
    [
        Output('data_racetrack','children'),
        Output('data_contour','children')
    ],
    Input('racetrack_dropdown_size','value')
)
def update_data(size):
    return [
        pd.DataFrame(np.load(os.path.join(DATA_PATH,f'racetrack{size}.npy')),columns=['b','X','y']).to_json(orient='split'),
        pd.DataFrame(np.load(os.path.join(DATA_PATH,f'contour{size}.npy'))).to_json(orient='split'),
    ]

@app.callback(
    Output('best_theta_racetrack','children'),
    Input('data_racetrack','children')
)
def best_theta(data):
    df = pd.read_json(data, orient='split')
    
    lr = LinearRegression().fit(df[['X']],df['y'])
    theta0=lr.intercept_
    theta1=lr.coef_[0]

    return [theta0, theta1]


@app.callback(
    [
        Output('bgd_history','children'),
        Output('sgd_history','children'),
        Output('mbgd_history','children'),
    ],
    [
        Input('racetrack_button_nextstep','n_clicks'),
        Input('racetrack_button_reset','n_clicks'),
        Input('best_theta_racetrack','children'),
        State('data_racetrack','children'),
        State('bgd_history','children'),
        State('sgd_history','children'),
        State('mbgd_history','children'),
        State('racetrack_input_steps','value'),
        State('bgd_lr_dropdown','value'),
        State('sgd_lr_dropdown','value'),
        State('mbgd_lr_dropdown','value'),
        State('mbgd_batchsize','value')
    ]
)
def update_history(n_clicks, n_clicks_reset, best_theta, data, bgd_history, sgd_history, mbgd_history, n_steps, bgd_lr, sgd_lr, mbgd_lr, mbgd_batchsize):
    if steps_check(n_steps, 1, 50)==False:
        raise PreventUpdate
    
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if 'best_theta' in ctx or 'reset' in ctx or ctx=='.':
        theta0, theta1 = 0, 0
        while 10>theta0>-10 or 10>theta1>-10:
            theta0 = round(random.uniform(best_theta[0]-15,best_theta[0]+15),2)
            theta1 = round(random.uniform(best_theta[1]-15,best_theta[1]+15),2)
        d = {'theta0':[theta0],'theta1':[theta1],'eta':[0.1],'epoch':[0],'batch':[0],'t':[ np.nan]}
        bgd_history = pd.DataFrame(d)
        sgd_history = pd.DataFrame(d)
        mbgd_history = pd.DataFrame(d)
    else:
        bgd_history = pd.read_json(bgd_history, orient='split')
        sgd_history = pd.read_json(sgd_history, orient='split')
        mbgd_history = pd.read_json(mbgd_history, orient='split')
        data = pd.read_json(data, orient='split').to_numpy()
        n = len(bgd_history)

        next_bgd = [bgd_history.values[-1]]
        next_sgd = [sgd_history.values[-1]]
        next_mbgd = [mbgd_history.values[-1]]
        
        for i in range(n_steps):
            next_bgd.append(bgd(data[:,:2],data[:,-1:],*next_bgd[-1][:4],lr_dict[bgd_lr]))
            next_sgd.append(sgd(data[:,:2],data[:,-1:],*next_sgd[-1][:5],lr_dict[sgd_lr]))
            next_mbgd.append(mbgd(data[:,:2],data[:,-1:],*next_mbgd[-1][:5],mbgd_batchsize,lr_dict[mbgd_lr]))
        
        bgd_history = bgd_history.append(pd.DataFrame(next_bgd[1:], columns=['theta0','theta1','eta','epoch','batch','t']), ignore_index=True)
        sgd_history = sgd_history.append(pd.DataFrame(next_sgd[1:], columns=['theta0','theta1','eta','epoch','batch','t']), ignore_index=True)
        mbgd_history = mbgd_history.append(pd.DataFrame(next_mbgd[1:], columns=['theta0','theta1','eta','epoch','batch','t']), ignore_index=True)
        
    return [
        bgd_history.to_json(orient='split'),
        sgd_history.to_json(orient='split'),
        mbgd_history.to_json(orient='split'),
    ]

@app.callback(
    Output('racetrack_graph_regression','figure'),
    [   
        Input('data_racetrack','children'),
        Input('bgd_history','children'),
        State('sgd_history','children'),
        State('mbgd_history','children'),
        State('racetrack_graph_regression','figure'),
        State('racetrack_dropdown_size','value')
    ]
)
def update_regression(data, bgd_history, sgd_history, mbgd_history, fig, size):
    ctx = dash.callback_context.triggered[0]['prop_id']

    data = pd.read_json(data, orient='split')
    bgd_history = pd.read_json(bgd_history, orient='split')
    last_bgd = bgd_history[['theta0','theta1']].values[-1].reshape(2,1)
    last_sgd = pd.read_json(sgd_history, orient='split')[['theta0','theta1']].values[-1].reshape(2,1)
    last_mbgd = pd.read_json(mbgd_history, orient='split')[['theta0','theta1']].values[-1].reshape(2,1)
    
    if len(bgd_history)==1:
        d = {
            100:100,
            500:500,
            1000:1000,
            5000:1500,
            10000:2000
        }
        figure = go.Figure(
            data=[
                go.Scatter(
                    x=data['X'][:d[size]],
                    y=data['y'][:d[size]],
                    mode='markers',
                    opacity=0.75,
                    name='Data',
                    hovertemplate='(%{x:.2f}, %{y:.2f})'
                )
            ],
            layout=go.Layout(
                yaxis = dict(
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="lightgrey",
                    gridcolor="lightgrey",
                    title='y',
                ),
                xaxis = dict(
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="lightgrey",
                    gridcolor="lightgrey",
                    title='X',
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
    else:
        figure = go.Figure(data=fig['data'], layout=fig['layout'])

    idx = [i for i,x in enumerate(figure['data']) if x.name in ['BGD','SGD','MBGD']]
    if idx != []:
        fdata = list(figure['data'])
        idx.sort(reverse=True)
        for i in idx:
            fdata.pop(i)
        figure['data'] = tuple(fdata)

    figure.add_trace(
        go.Scatter(x=data['X'],y=(data[['b','X']].to_numpy().dot(last_bgd).ravel()), mode='lines', line=dict(color='red'), name='BGD', hovertemplate='(%{x:.2f}, %{y:.2f})'),
    )
    figure.add_trace(
        go.Scatter(x=data['X'],y=(data[['b','X']].to_numpy().dot(last_sgd).ravel()), mode='lines', line=dict(color='lime'), name='SGD', hovertemplate='(%{x:.2f}, %{y:.2f})'), 
    )
    figure.add_trace(
        go.Scatter(x=data['X'],y=(data[['b','X']].to_numpy().dot(last_mbgd)).ravel(), mode='lines', line=dict(color='gold'), name='MBGD', hovertemplate='(%{x:.2f}, %{y:.2f})'), 
    )
    
    return figure

@app.callback(
    Output('racetrack_graph_contour','figure'),
    [
        Input('data_contour','children'),
        Input('bgd_history','children'),
        State('sgd_history','children'),
        State('mbgd_history','children'),
        State('racetrack_graph_contour','figure'),
        State('best_theta_racetrack','children')
    ]
)
def update_contour(data_contour, bgd_history, sgd_history, mbgd_history, fig, best_theta):
    ctx = dash.callback_context.triggered[0]['prop_id']

    bgd_history = pd.read_json(bgd_history, orient='split')
    sgd_history = pd.read_json(sgd_history, orient='split')
    mbgd_history = pd.read_json(mbgd_history, orient='split')
    
    if len(bgd_history)==1:
        data_contour = pd.read_json(data_contour, orient='split').to_numpy()
        theta0_axis = np.linspace(best_theta[0]-15,best_theta[0]+15,90)
        theta1_axis = np.linspace(best_theta[1]-15,best_theta[1]+15,90)

        figure = go.Figure(
            data=[
                go.Contour(
                    x=theta0_axis,
                    y=theta1_axis,
                    z=data_contour,
                    contours=dict(
                        coloring ='heatmap',
                        showlabels = True,
                        labelfont = dict(
                            size = 8,
                            color = 'white',
                        ),
                    ),
                    name='Cost function',
                    hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}<br>MSE cost = %{z:.2f}'
                ),
                go.Scatter(
                    x=[best_theta[0]],
                    y=[best_theta[1]],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        color='limegreen',
                        size=8
                    ),
                    name='Global optimum',
                    showlegend=False,
                    hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}'
                ),
                go.Scatter(
                    x=[bgd_history['theta0'].values[-1]],
                    y=[bgd_history['theta1'].values[-1]],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color='red',
                        size=8
                    ),
                    name='Random initialization',
                    showlegend=False,
                    hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}'
                ),
            ],
            layout=go.Layout(
                    yaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='theta_1',
                        range=[best_theta[1]-15,best_theta[1]+15]
                    ),
                    xaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='theta_0',
                        range=[best_theta[0]-15,best_theta[0]+15]
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
    else:
        figure = go.Figure(data=fig['data'], layout=fig['layout'])
    
    idx = [i for i,x in enumerate(figure['data']) if x.name in ['BGD','SGD','MBGD']]
    
    if idx != []:
        fdata = list(figure['data'])
        idx.sort(reverse=True)
        for i in idx:
            fdata.pop(i)
        figure['data'] = tuple(fdata)

    figure.add_trace(
        go.Scatter(x=bgd_history['theta0'],y=bgd_history['theta1'], name='BGD', mode='lines', line=dict(color='red'), opacity=0.8, showlegend=False, hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}')
    )
    figure.add_trace(
        go.Scatter(x=sgd_history['theta0'],y=sgd_history['theta1'], name='SGD', mode='lines', line=dict(color='lime'), opacity=0.8, showlegend=False, hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}')
    )
    figure.add_trace(
        go.Scatter(x=mbgd_history['theta0'],y=mbgd_history['theta1'], name='MBGD', mode='lines', line=dict(color='gold'), opacity=0.8, showlegend=False, hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}')
    )
    
    return figure
        
@app.callback(
    [
        Output('bgd_batchsize','children'),
        Output('bgd_epoch','children'),
        Output('bgd_batch','children'),
        Output('bgd_learningrate','children'),
        Output('bgd_mse','children'),
        Output('bgd_mse_rank','children'),
        Output('bgd_time','children'),
        Output('bgd_time_rank','children'),

        Output('sgd_batchsize','children'),
        Output('sgd_epoch','children'),
        Output('sgd_batch','children'),
        Output('sgd_learningrate','children'),
        Output('sgd_mse','children'),
        Output('sgd_mse_rank','children'),
        Output('sgd_time','children'),
        Output('sgd_time_rank','children'),

        Output('mbgd_epoch','children'),
        Output('mbgd_batch','children'),
        Output('mbgd_learningrate','children'),
        Output('mbgd_mse','children'),
        Output('mbgd_mse_rank','children'),
        Output('mbgd_time','children'),
        Output('mbgd_time_rank','children'),
    ],
    [
        Input('bgd_history','children'),
        State('sgd_history','children'),
        State('mbgd_history','children'),
        State('racetrack_dropdown_size','value'),
        State('data_racetrack','children')
    ]
)
def update_trs(bgd_history, sgd_history, mbgd_history, size, data):
    data = pd.read_json(data, orient='split')
    bgd = pd.read_json(bgd_history, orient='split')
    sgd = pd.read_json(sgd_history, orient='split')
    mbgd = pd.read_json(mbgd_history, orient='split')
    
    bgd_theta = np.array([[bgd.iloc[-1]['theta0']],[bgd.iloc[-1]['theta1']]])
    sgd_theta = np.array([[sgd.iloc[-1]['theta0']],[sgd.iloc[-1]['theta1']]])
    mbgd_theta = np.array([[mbgd.iloc[-1]['theta0']],[mbgd.iloc[-1]['theta1']]])

    bgd_mse = round(mean_squared_error(data['y'],data[['b','X']].to_numpy().dot(bgd_theta)),2)
    sgd_mse = round(mean_squared_error(data['y'],data[['b','X']].to_numpy().dot(sgd_theta)),2)
    mbgd_mse = round(mean_squared_error(data['y'],data[['b','X']].to_numpy().dot(mbgd_theta)),2)
    mse_ranks = stats.rankdata([bgd_mse,sgd_mse,mbgd_mse], method='dense')
    
    bgd_avgt = bgd['t'].mean()
    sgd_avgt = sgd['t'].mean()
    mbgd_avgt = mbgd['t'].mean()
    times = [bgd_avgt,sgd_avgt,mbgd_avgt]
    if all(np.isnan(t) for t in times):
        t_ranks = [1,1,1]
        bgd_avgt,sgd_avgt,mbgd_avgt=0,0,0
    else:
        t_ranks = stats.rankdata(times, method='dense')

    return [
        size,
        bgd.iloc[-1]['epoch'],
        f"{min(1,int(bgd.iloc[-1]['batch']))}/1",
        round(bgd.iloc[-1]['eta'],4),
        bgd_mse,
        ranks_dict[mse_ranks[0]],
        [int(bgd_avgt),'$\\mu s$'],
        ranks_dict[t_ranks[0]],

        1,
        sgd.iloc[-1]['epoch'],
        f"{int(sgd.iloc[-1]['batch'])}/10",
        round(sgd.iloc[-1]['eta'],4),
        sgd_mse,
        ranks_dict[mse_ranks[1]],
        [int(sgd_avgt),'$\\mu s$'],
        ranks_dict[t_ranks[1]],

        mbgd.iloc[-1]['epoch'],
        f"{int(mbgd.iloc[-1]['batch'])}/10",
        round(mbgd.iloc[-1]['eta'],4),
        mbgd_mse,
        ranks_dict[mse_ranks[2]],
        [int(mbgd_avgt),'$\\mu s$'],
        ranks_dict[t_ranks[2]],
    ]

@app.callback(
    Output('mbgd_batchsize_td','children'),
    [
        Input('racetrack_dropdown_size','value'),
        Input('racetrack_button_nextstep','n_clicks'),
        Input('racetrack_button_reset','n_clicks'),
        Input('racetrack_href_li1_1','n_clicks'),
        Input('racetrack_href_li1_2','n_clicks'),
        Input('racetrack_href_li2_1','n_clicks'),
        Input('racetrack_href_li2_2','n_clicks'),
        State('mbgd_batchsize_td','children')
    ]
)
def update_mbgd_batchsize(size, nextstep, reset, li1_1, li1_2, li2_1, li2_2, mbgd_batchsize_td):
    ctx = dash.callback_context.triggered[0]['prop_id']

    def create_dropdown(size, value):
        return dcc.Dropdown(
                id='mbgd_batchsize',
                options=[
                    {'label':1, 'value':1},
                    {'label':size*0.02, 'value':size*0.02},
                    {'label':size*0.2, 'value':size*0.2},
                    {'label':size*0.5, 'value':size*0.5},
                    {'label':size*0.7, 'value':size*0.7},
                    {'label':size, 'value':size},
                ],
                value=value,
                disabled=False,
                clearable=False,
                searchable=False,
            )
    

    if ctx=='.' or 'reset' in ctx or 'size' in ctx:
        return create_dropdown(size,size*0.2)
    elif 'li1_1' in ctx:
        return create_dropdown(10000,200)
    elif 'li2_1' in ctx:
        return create_dropdown(100,2)
    elif 'li2_2' in ctx or 'li1_2' in ctx:
        return create_dropdown(100,70)
    else:
        mbgd_batchsize_td['props']['disabled'] = True
        return mbgd_batchsize_td

@app.callback(
    [
        Output('racetrack_dropdown_size','disabled'),
        Output('bgd_lr_dropdown','disabled'),
        Output('sgd_lr_dropdown','disabled'),
        Output('mbgd_lr_dropdown','disabled'),
    ],
    [
        Input('racetrack_button_nextstep','n_clicks'),
        Input('racetrack_button_reset','n_clicks'),
    ]
)
def disable_options(nextstep, reset):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx=='.' or 'reset' in ctx:
        return [False,False,False,False]
    else:
        return [True,True,True,True]

@app.callback(
    [
        Output('racetrack_dropdown_size','value'),
        Output('racetrack_input_steps','value'),
        Output('bgd_lr_dropdown','value'),
        Output('sgd_lr_dropdown','value'),
        Output('mbgd_lr_dropdown','value'),
    ],
    [
        Input('racetrack_href_li1_1','n_clicks'),
        Input('racetrack_href_li1_2','n_clicks'),
        Input('racetrack_href_li2_1','n_clicks'),
        Input('racetrack_href_li2_2','n_clicks'),
        State('racetrack_dropdown_size','value'),
        State('racetrack_input_steps','value'),
        State('bgd_lr_dropdown','value'),
        State('sgd_lr_dropdown','value'),
        State('mbgd_lr_dropdown','value'),
    ]
)
def hrefs(li1_1, li1_2, li2_1, li2_2, racetrack_dropdown_size, racetrack_input_steps, bgd_lr_dropdown, sgd_lr_dropdown, mbgd_lr_dropdown):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if 'li1_1' in ctx:
        return [10000,5,'constant','time','time']
    elif 'li2_1' in ctx:
        return [100,5,'constant','time','time']
    elif 'li2_2' in ctx or 'li1_2' in ctx:
        return [100,5,'constant','time','time']
    else:
        return [racetrack_dropdown_size, racetrack_input_steps, bgd_lr_dropdown, sgd_lr_dropdown, mbgd_lr_dropdown]