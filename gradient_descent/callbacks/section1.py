import dash
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import dash_html_components as html
import dash_bootstrap_components as dbc

from sklearn.metrics import mean_squared_error
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from util import j, djt1, array2matrix
from app import app


def eta_check(eta, min, max):
    if eta is None or not min<=eta<=max:
        return False
    else:
        return True

@app.callback(Output('dashboard1_p_theta1','children'),Input('dashboard1_slider_theta1','value'))
def dashboard1_p_theta0(theta1):
    return ['$\\theta_1=$',' ',theta1]

@app.callback(Output('dashboard1_slider_theta1','value'),[Input('dashboard1_button_bestfit','n_clicks'),State('dashboard1_slider_theta1','value'),State('best_theta','children')])
def best_fit_dashboard1(n_clicks,theta1,best_theta):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.':
        return theta1
    else:
        return round(best_theta[1],1)

@app.callback(Output('dashboard1_div_mse','children'),[Input('dashboard1_slider_theta1','value'),State('data','children'),State('best_theta','children')])
def dashboard1_div_mse(theta1,data,best_theta):
    df = pd.read_json(data, orient='split')
    mse = mean_squared_error(df['y'], theta1*df['X']+best_theta[0])
    
    return [html.H3('MSE'),f'{mse:.2f}']

@app.callback(Output('dashboard1_graph_regression','figure'),[Input('dashboard1_slider_theta1','value'),Input('dashboard1_checklist_residuals','value'),State('data','children'), State('best_theta','children')])
def dashboard1_graph_regression(theta1, residuals_switch, data, best_theta):
    df = pd.read_json(data, orient='split')
    
    fig = go.Figure(
        data=[
            go.Scatter(x=df['X'],y=df['y'],mode='markers',marker=dict(color='firebrick'),name='Data', hovertemplate='(%{x:.2f}, %{y:.2f})'),
            go.Scatter(x=df['X'],y=theta1*df['X']+best_theta[0],mode='lines',line=dict(color='royalblue'),name='Prediction', hovertemplate='(%{x:.2f}, %{y:.2f})')
        ],
        layout = go.Layout(
            yaxis = dict(
                mirror=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="lightgrey",
                gridcolor="lightgrey",
                title='y',
                range = [0,13.5]
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
    
    if residuals_switch != []:
        preds = theta1*df['X']+best_theta[0]
        residuals = df['y'] - preds
        for i, r in enumerate(residuals):
            fig.add_trace(
                go.Scatter(
                    x=[df['X'][i], df['X'][i]],
                    y=[preds[i], preds[i] + r],
                    line=dict(color="firebrick", width=1, dash="dash"),
                    name=f'Residual {i+1}',
                    showlegend=False,
                    hovertemplate='(%{x:.2f}, %{y:.2f})'
                )
            )
        
        fig.add_trace(
            go.Scatter(
                x=[-100, -100],
                y=[-100, -100],
                name="Residuals",
                line=dict(color="firebrick", width=1, dash="dash"),
            )
        )
    
    return fig

@app.callback(Output('dashboard2_div_mse','children'),[Input('dashboard2_div_theta1init','children'),Input('dashboard2_div_theta1hist','children'), State('data','children'), State('best_theta','children')])
def dashboard2_div_mse(theta1_init, theta1_hist, data, best_theta):
    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context
    
    if ctx.triggered[0]['prop_id'] == 'theta1_init.children':
        theta1 = theta1_init
    else:
        theta1 = theta1_hist[-1]
    
    mse = j(df[['b','X']].to_numpy(),df[['y']].to_numpy(),best_theta[0],theta1)
        
    return [html.H3('MSE'),f'{mse:.2f}']
    
@app.callback(
    [Output('dashboard2_div_theta1init','children'),
    Output('dashboard2_input_eta','value')],
    [Input('dashboard2_button_reset','n_clicks'),
     Input('dashboard2_href_li1','n_clicks'),
     Input('dashboard2_href_li2','n_clicks'),
     Input('dashboard2_href_li3','n_clicks'),
     Input('dashboard2_href_li4','n_clicks'),
     State('best_theta','children'),
     State('dashboard2_input_eta','value')]
     )
def dashboard2_theta1init(n_clicks,li1,li2,li3,li4,best_theta,eta):
    if eta_check(eta, 0, 2)==False:
        raise PreventUpdate

    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if 'reset' in ctx:
        v = best_theta[1]
        while best_theta[1]+1.5>v>best_theta[1]-1.5:
            v = round(random.uniform(-1.5,7.5),2)
    elif 'li1' in ctx:
        return [4.9,1.5]
    elif 'li2' in ctx:
        return [1.8,2]
    elif 'li3' in ctx:
        return [-1,0.05]
    elif 'li4' in ctx:
        return [7.5,0.4]
    else:
        v = 6
                
    return [v,eta]

@app.callback(
    Output('dashboard2_input_eta','disabled'),
    [Input('dashboard2_button_reset','n_clicks'),
     Input('dashboard2_button_nextstep','n_clicks'),
     Input('dashboard2_button_nextstep_table','n_clicks'),
     State('dashboard2_input_eta','value')]
)
def dashboard2_input_eta(reset,nextstep,nextstep_table,eta):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if 'reset' in ctx or all(v is None for v in [nextstep,nextstep_table]) or eta_check(eta, 0 ,2)==False:
        return False
    else:
        return True

@app.callback(
    [Output('dashboard2_div_theta1hist','children'),
     Output('dashboard2_div_thetas','children')],
    [Input('dashboard2_button_nextstep','n_clicks'),
     Input('dashboard2_button_nextstep_table','n_clicks'),
     Input('dashboard2_div_theta1init','children'),
     State('dashboard2_div_theta1hist','children'),
     State('dashboard2_input_eta','value'),
     State('best_theta','children'),
     State('data','children')]
)
def dashboard2_theta1hist(n_clicks,n_clicks_table,theta1_init,theta1_hist,eta,theta_best,data):
    if eta_check(eta, 0, 2)==False:
        raise PreventUpdate

    df = pd.read_json(data, orient='split')
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.' or 'theta1init' in ctx:
        return [[theta1_init], ['$\\theta_0=$',' ',round(theta_best[0],1),', ','$\\theta_1=$',' ',round(theta1_init,1)]]
    else:
        gradient = djt1(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta_best[0],theta1_hist[-1])
        theta1_new = theta1_hist[-1] - eta * gradient
        return [theta1_hist+[theta1_new], ['$\\theta_0=$',' ',round(theta_best[0],1),', ','$\\theta_1=$',' ',round(theta1_new,1)]]

@app.callback(
    Output('dashboard2_table_math','children'),
    [Input('dashboard2_div_theta1hist','children'),
     Input('dashboard2_input_eta','value'),
     State('dashboard2_table_math','children'),
     State('best_theta','children'),
     State('data','children')]
)
def dashboard2_table_math(theta1hist, eta, table, best_theta, data):
    if eta_check(eta, 0, 2)==False:
            raise PreventUpdate

    df = pd.read_json(data, orient='split')
    theta0 = round(best_theta[0],2)
    td_style = {'text-align':'center','vertical-align':'middle','font-weight':'bold'}
    if len(theta1hist)==1:
        theta1 = round(theta1hist[-1],2)
        theta = np.array([[theta0],[theta1]])
        trs = [
            html.Tr([
                html.Td('Parameters', style=td_style),
                html.Td(f'$$\\theta=\\begin{{pmatrix}}\\theta_0^{{best}}\\\\\\theta_1\\end{{pmatrix}}=\\begin{{pmatrix}}{theta0}\\\\\\theta_1\\end{{pmatrix}}$$$$\\eta={eta}$$')
            ]),
            html.Tr([
                html.Td('Cost function', style=td_style),
                html.Td(f'$$\\begin{{aligned}}\\textrm{{MSE}}(\\theta)&=\\frac{{1}}{{2m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})^{{2}}\\\\&=\\frac{{1}}{{2m}}\\sum_{{i=1}}^{{m}}(\\begin{{pmatrix}}{theta0}&\\theta_{{1}}\\end{{pmatrix}}x^{{i}}-y^{{i}})^{{2}}\\end{{aligned}}$$')
            ]),
            html.Tr([
                html.Td('Gradient', style=td_style),
                html.Td(f'$$\\begin{{aligned}}\\frac{{\\partial}}{{\\partial{{\\theta_1}}}}\\textrm{{MSE}}(\\theta)&=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\theta^{{T}}x^{{i}}-y^{{i}})x_{{1}}^{{i}}\\\\&=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}(\\begin{{pmatrix}}{theta0}&\\theta_{{1}}\\end{{pmatrix}}x^{{i}}-y^{{i}})x_{{1}}^{{i}}\\end{{aligned}}$$')
            ]),
            html.Tr([
                html.Td('Initialization', style=td_style),
                html.Td([
                    f'$$\\theta={array2matrix(theta)}$$',
                    ])
            ])
        ]
        return html.Tbody(trs, className='math-left-align')
    else:
        theta1_new = round(theta1hist[-1],2)
        theta1_old = round(theta1hist[-2],2)
        theta = np.array([[theta0],[theta1_old]])
        gradient = djt1(df[['b','X']].to_numpy(),df[['y']].to_numpy(),theta0,theta1_old)
        trs = table['props']['children']
        tr = html.Tr(
            [
                html.Td(f'Step {len(theta1hist)-1}', style=td_style),
                html.Td([
                    f'$$\\frac{{\\partial}}{{\\partial{{\\theta_1}}}}\\textrm{{MSE}}(\\theta)=\\frac{{1}}{{m}}\\sum_{{i=1}}^{{m}}({array2matrix(theta.T)}x^{{i}}-y^{{i}})x_{{1}}^{{i}}={round(gradient,2)}$$',
                    f'$$\\theta_{{1}}^{{new}}=\\theta_1-\\eta\\textrm{{MSE}}_{{\\theta_1}}(\\theta)={theta1_old}-{eta}\\cdot{round(gradient,2)}={theta1_new}$$',
                    f'$$\\theta={array2matrix(np.array([[theta0],[theta1_new]]))}$$'
                
                ])
            ])
        return html.Tbody(trs+[tr], className='math-left-align')

@app.callback(
    Output('dashboard2_graph_regression', 'figure'),
    [Input('dashboard2_div_theta1init','children'),
     Input('dashboard2_div_theta1hist','children'),
     State('dashboard2_graph_regression','figure'),
     State('data','children'),
     State('best_theta','children')]
)
def dashboard2_regression(theta1_init, theta1_hist, fig, data, best_theta):
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
    Output('dashboard2_graph_lossfunction', 'figure'),
    [Input('dashboard2_div_theta1hist','children'),
     Input('dashboard2_div_theta1init','children'),
     State('dashboard2_graph_lossfunction','figure'),
     State('data','children'),
     State('best_theta','children')]
)
def dashboard2_lossfunction(theta1_hist, theta1_init, fig, data, best_theta):
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
