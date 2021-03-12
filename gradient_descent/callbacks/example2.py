import dash
import pandas as pd
import plotly.graph_objects as go
import dash_html_components as html

from sklearn.metrics import mean_squared_error
from dash.dependencies import Input, Output, State
from app import app


@app.callback(Output('example2_p_theta0','children'),Input('example2_slider_theta0','value'))
def p_theta0_example2(theta0):
    return ['theta_0 = ',theta0]

@app.callback(Output('example2_p_theta1','children'),Input('example2_slider_theta1','value'))
def p_theta1_example2(theta1):
    return ['theta_1 = ',theta1]

@app.callback([Output('example2_slider_theta0','value'),Output('example2_slider_theta1','value')],[Input('example2_button_bestfit','n_clicks'),State('example2_slider_theta0','value'),State('example2_slider_theta1','value'),State('best_theta','children')])
def best_fit_example2(n_clicks,theta0,theta1,best_theta):
    ctx = dash.callback_context.triggered[0]['prop_id']
    
    if ctx == '.':
        return [theta0,theta1]
    else:
        return [round(best_theta[0],1),round(best_theta[1],1)]

@app.callback(Output('example2_div_mse','children'),[Input('example2_slider_theta0','value'),Input('example2_slider_theta1','value'),State('data','children')])
def example2_div_mse(theta0, theta1, data):
    df = pd.read_json(data, orient='split')
    mse = mean_squared_error(df['y'], theta1*df['X']+theta0)
    
    return [html.H3('MSE'),f'{mse:.2f}']

@app.callback(Output('example2_graph_regression','figure'),[Input('example2_slider_theta0','value'),Input('example2_slider_theta1','value'),Input('example2_checklist_residuals','value'),State('data','children'), State('best_theta','children')])
def graph_example2(theta0, theta1, residuals_switch, data, best_theta):
    df = pd.read_json(data, orient='split')
        
    fig = go.Figure(
        data=[
            go.Scatter(x=df['X'],y=df['y'],mode='markers',marker=dict(color='firebrick'),name='Data', hovertemplate='(%{x:.2f}, %{y:.2f})'),
            go.Scatter(x=df['X'],y=theta1*df['X']+theta0,mode='lines',line=dict(color='royalblue'),name='Prediction', hovertemplate='(%{x:.2f}, %{y:.2f})')
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
        preds = theta1*df['X']+theta0
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