import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
# import chart_studio.plotly as py
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import docx
import datetime as dt
import pickle
import random
import math
import time
import os
import re
import colorsys
import arrow
import locale
import requests
import collections
import numbers
import decimal

from warnings import simplefilter

from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from dash import Dash, html, dcc, callback, Output, Input
from __functions import *

path_current = os.getcwd()
path_files = path_current + '/files'
economics_data = loadit('economics_data', path=path_files, create_empty_dict=True)
prices_food_growth = economics_data['prices_food_growth'].copy()

df1 = smoothed(prices_food_growth, datetime_index=True)
df2 = prices_food_growth.copy()

markdown_text = '''
    Статистические данные об относительном изменении цен с января 2021 года
'''

app = Dash()

app.layout = [
    html.H1(children='Динамика цен на продукты питания', style={'textAlign':'left', 'font-family': 'Ubuntu'}),
    html.H2(children=markdown_text, style={'textAlign':'right'}),
    dcc.Dropdown(options=df1.columns, value='овощи', id='dropdown-selection', style={'textAlign':'left', 'font-family': 'Ubuntu'}),
    dcc.Graph(id='graph-content', style={'width': '100%', 'height': '100%'})]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value'))

def update_graph(value):
    
    df3 = df1[value]
    df4 = df2[value]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df3.index,
            y=df3,
            marker=dict(opacity=0, size=6),
            mode='lines', line_color=palette[1], hoverinfo='skip',
            name=value.capitalize()))

    fig.add_trace(
        go.Scatter(
            x=df4.index,
            y=df4,
            marker=dict(opacity=0, size=6),
            mode='markers', line_color=palette[1], showlegend=False,
            name=value.capitalize()))

    fig.update_xaxes(dtick='M3', tickformat="%b<br>%Y", labelalias=xtickaliases)
    fig.update_yaxes(dtick=0.1)
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)

