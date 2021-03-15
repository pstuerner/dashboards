import dash
import pandas as pd
import numpy as np
import sympy as sp
import random
import plotly.graph_objects as go
import dash_html_components as html

from sklearn.metrics import mean_squared_error
from dash.dependencies import Input, Output, State
from util import j, djt0, djt1, z
from app import app


grid_lf = np.linspace(-20,20,90)
x_lf, y_lf = np.meshgrid(grid_lf,grid_lf)
colors_plane = np.zeros(shape=(90,90))
colorscale = [[0, 'blue'], [1, 'blue']]


@app.callback(
    Output('example4_div_mse','children'),
    [Input('example4_div_thetainit','children'),
     Input('example4_div_thetahist','children'),
     Input('example4_checklist_scalex','value'),
     State('data','children'),
     State('data_scaled','children')]
)
def example4_div_mse(theta_init, theta_hist, scalex, data, data_scaled):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'thetainit' in ctx:
        theta0, theta1 = theta_init
    else:
        df_theta = pd.read_json(theta_hist, orient='split')
        theta0, theta1 = df_theta.values[-1]
        
        if scalex != []:
            df = pd.read_json(data_scaled, orient='split')
        else:
            df = pd.read_json(data, orient='split')
    
    mse = j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta0,theta1)
        
    return [html.H3('MSE'),f'{mse:.2f}']

@app.callback(Output('example4_div_thetainit','children'),[Input('example4_button_reset','n_clicks'),Input('example4_checklist_scalex','value')])
def example4_thetainit(n_clicks, scalex):
    ctx = dash.callback_context
    
    if 'reset' in ctx.triggered[0]['prop_id'] or scalex != []:
        theta0_init, theta1_init = 0, 0
        while 10>theta0_init>-10 or 10>theta1_init>-10:
            theta0_init = round(random.uniform(-20,20),2)
            theta1_init = round(random.uniform(-20,20),2)
    else:
        theta0_init, theta1_init = -10, -10
                
    return [theta0_init, theta1_init]

@app.callback(
    [Output('example4_div_thetahist','children'),
     Output('example4_div_thetas','children')],
    [Input('example4_button_nextstep','n_clicks'),
     Input('example4_div_thetainit','children'),
     State('example4_checklist_scalex','value'),
     State('example4_div_thetahist','children'),
     State('example4_input_eta','value'),
     State('data','children'),
     State('data_scaled','children')]
)
def example4_thetahist(n_clicks,theta_init,scalex,theta_hist,eta,data,data_scaled):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'thetainit' in ctx:
        theta0, theta1 = theta_init
        df_theta = pd.DataFrame({'theta0':[theta0],'theta1':[theta1]})
        return [df_theta.to_json(orient='split'), ['$\\theta_0=$',' ',round(theta0,1),', ','$\\theta_1=$',' ',round(theta1,1)]]
    else:
        df_theta = pd.read_json(theta_hist, orient='split')
        theta0, theta1 = df_theta.values[-1]
        
        if scalex != []:
            df = pd.read_json(data_scaled, orient='split')
        else:
            df = pd.read_json(data, orient='split')

        dt0 = djt0(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta0,theta1)
        dt1 = djt1(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta0,theta1)
        theta0_new = theta0 - eta * dt0
        theta1_new = theta1 - eta * dt1
        df_theta = df_theta.append({'theta0':theta0_new,'theta1':theta1_new}, ignore_index=True)

        return [df_theta.to_json(orient='split'), ['$\\theta_0=$',' ',round(theta0_new,1),', ','$\\theta_1=$',' ',round(theta1_new,1)]]

@app.callback(
    Output('example4_graph_regression', 'figure'),
    [Input('example4_div_thetainit','children'),
     Input('example4_div_thetahist','children'),
     State('example4_checklist_scalex','value'),
     State('example4_graph_regression','figure'),
     State('data','children'),
     State('data_scaled','children')]
)
def example4_regression(theta_init, theta_hist, scalex, fig, data, data_scaled):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if scalex != []:
        df = pd.read_json(data_scaled, orient='split')
    else:
        df = pd.read_json(data, orient='split')
    
    if ctx == '.' or 'thetainit' in ctx:
        theta0_init, theta1_init = theta_init
        t_ = np.array([[theta0_init],[theta1_init]])
        return go.Figure(
                data=[
                    go.Scatter(x=df['X'], y=df['y'], mode='markers', name='Data', hovertemplate='(%{x:.2f}, %{y:.2f})'),
                    go.Scatter(x=df['X'], y=theta0_init+theta1_init*df['X'], name='Initialization', line = dict(color='red', width=2, dash='dot'), hovertemplate='(%{x:.2f}, %{y:.2f})')              
                ],
                layout=go.Layout(
                    yaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='y',
                        range=[min(-1,df[['b','X']].to_numpy().dot(t_).min())/0.9,max(15,df[['b','X']].to_numpy().dot(t_).max())/0.9]
                    ),
                    xaxis = dict(
                        mirror=True,
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor="lightgrey",
                        gridcolor="lightgrey",
                        title='X',
                        range = [0,df['X'].max()*1.1]
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
    elif 'thetahist' in ctx:
        df_theta = pd.read_json(theta_hist, orient='split')
        showlegend = True if len(df_theta)==2 else False
        theta0, theta1 = df_theta.values[-1]
        t_ = np.array([[theta0],[theta1]])
        
        figure = go.Figure(data=fig['data'], layout=fig['layout'])
        figure.add_trace(go.Scatter(x=df['X'], y=theta0+theta1*df['X'], mode='lines', line=dict(color='black'), name=f'Steps', showlegend=showlegend, hovertemplate='(%{x:.2f}, %{y:.2f})'))
        figure['layout']['yaxis']['range'] = [min(-1,df[['b','X']].to_numpy().dot(t_).min())/0.9,max(15,df[['b','X']].to_numpy().dot(t_).max())/0.9]
        
        return figure

@app.callback(
    Output('example4_graph_lossfunction', 'figure'),
    [Input('example4_div_thetainit','children'),
     Input('example4_div_thetahist','children'),
     State('example4_checklist_scalex','value'),
     State('example4_graph_lossfunction','figure'),
     State('data','children'),
     State('data_scaled','children')]
)
def example4_lossfunction(theta_init, theta_hist, scalex, fig, data, data_scaled):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if scalex != []:
        df = pd.read_json(data_scaled, orient='split')
    else:
        df = pd.read_json(data, orient='split')
    
    if ctx == '.' or 'thetainit' in ctx:
        theta0, theta1 = theta_init
        df_theta = pd.DataFrame({'theta0':[theta0],'theta1':[theta1]})
    else:
        df_theta = pd.read_json(theta_hist, orient='split')
        theta0, theta1 = df_theta.values[-1]
    
    t0,t1 = sp.symbols('t0,t1')
    loss_surface = sp.lambdify((t0,t1), j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),t0,t1))
    tangent_plane = sp.lambdify((t0,t1), z(t0,t1,theta0,theta1,df[['b','X']].to_numpy(),df[['y']].to_numpy()))

    grid_plane_theta0 = np.linspace(theta0-10,theta0+10,90)
    grid_plane_theta1 = np.linspace(theta1-10,theta1+10,90)
    x_plane, y_plane = np.meshgrid(grid_plane_theta0,grid_plane_theta1)
        
    if ctx == '.' or 'thetainit' in ctx:
        figure = go.Figure(
            data=[
                go.Surface(
                    z=loss_surface(x_lf,y_lf),
                    x=x_lf,
                    y=y_lf,
                    opacity=0.95,
                    colorscale='viridis',
                    cmin=0,
                    cmax=600,
                    name='Cost function',
                    showscale=False,
                    lighting=dict(ambient=0.8, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2),
                    hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}<br>MSE cost = %{z:.2f}'
                )
            ]
        )

        figure.update_traces(
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            )
        )
        
        figure.update_layout(
            scene = dict(
                xaxis = dict(
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="lightgrey",
                    gridcolor="lightgrey",
                    title='theta_0',
                    range=[-25,25]
                ),
                yaxis = dict(
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="lightgrey",
                    gridcolor="lightgrey",
                    title='theta_1',
                    range=[-25,25]
                ),
                zaxis = dict(
                    mirror=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="lightgrey",
                    gridcolor="lightgrey",
                    title='MSE Cost',
                    range=[-150,600]
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            uirevision=True,
            margin=dict(t=5,b=5,l=5,r=5),
        )
    else:
        figure = go.Figure(data=fig['data'], layout=fig['layout'])
    
    
    idx = [i for i,x in enumerate(figure['data']) if x.name in ['Gradient','Steps','Steps Contour']]
    if idx != []:
        fdata = list(figure['data'])
        idx.sort(reverse=True)
        for i in idx:
            fdata.pop(i)
        figure['data'] = tuple(fdata)
    
    figure.add_trace(
        go.Scatter3d(
            x=df_theta['theta0'],
            y=df_theta['theta1'],
            z=[j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta0,theta1) for theta0,theta1 in list(df_theta.to_records(index=False))],
            mode='lines+markers',
            line=dict(color='#FF7F0E', width=4),
            marker=dict(color='#FF7F0E'),
            name=f'Steps',
            showlegend=False,
            hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}<br>MSE cost = %{z:.2f}'
        )
    )
    
    figure.add_trace(
        go.Scatter3d(
            x=df_theta['theta0'],
            y=df_theta['theta1'],
            z=[-150]*len(df_theta),
            mode='lines+markers',
            line=dict(color='#FF7F0E', width=4),
            marker=dict(color='#FF7F0E'),
            name=f'Steps Contour',
            showlegend=False,
            hovertemplate='theta_0 = %{x:.2f}<br>theta_1 = %{y:.2f}'
        )
    )
    
    figure.add_trace(
        go.Surface(
            cmin=0,
            cmax=1,
            colorscale=colorscale,
            surfacecolor=colors_plane,
            z=tangent_plane(x_plane,y_plane)*.95,
            x=x_plane,
            y=y_plane,
            showscale=False,
            name='Gradient',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    return figure
