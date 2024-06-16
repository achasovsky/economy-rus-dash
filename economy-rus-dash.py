# // --- Import libraries --- //

# import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import plotly.graph_objects as go
# import plotly.express as px
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
import dash_bootstrap_components as dbc
# import scipy
# import statsmodels
# import statsmodels.api as sm
# import statsmodels.stats.api as sms
# import statsmodels.formula.api as smf
# import chart_studio.plotly as py
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import docx
# import datetime as dt
# import pickle
# import random
# import math
# import time
# import os
# import re
# import colorsys
# import arrow
# import locale
# import requests
# import collections
# import numbers
# import decimal

# from warnings import simplefilter

# from scipy import stats
# from scipy.interpolate import make_interp_spline, BSpline
# from sklearn import preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib.lines import Line2D
# import matplotlib.dates as mdates
from dash import Dash, html, dcc, callback, Output, Input
from functions import *


# // --- PLOTLY CONFIGURATION --- //


# font_family = 'PTSans'
# font_family = 'Avenir'
# font_family = 'NeverMindCompact'
# font_family = 'RG-StandardBook'
# font_family = 'TaylorSans'
# font_family =  'Tahoma'
# font_family =  'Trebuchet MS'
# font_family =  'Verdana'
font_family = 'Geologica Roman'
# font_family = 'PitagonSansMono'
# font_family = 'Roberto Sans'
# font_family = 'Vela Sans'
# font_family = 'Winston'



# font_family = 'system-ui'

pio.templates['primetheme'] = go.layout.Template(
    layout = {
        # 'dragmode': 'zoom',
        # 'hoverdistance': 0,
        'hidesources': True,
        # 'autosize': False,
        'margin': dict(
            t=20, r=35, l=50, b=35
        ),
        'font' : {
            'family': font_family,
            # 'family': 'Ubuntu',
            'size': 11.5,
        },
        'title': {
            'automargin': False,
            'pad': {'t': 10},
            'font': {
                'color': '#404040',
                'size': 15
            },
            'x': 0.1,
            'y': 1
        },
        'colorway': list(
            (
            #    0 : 0.3    1 red      2 blue     3 green    4 yellow
                '#4F4F4F', '#AF403D', '#31629A', '#429C7F', '#E3CB5F',
            #    5 brown    6 khaki    7 purple   8 pink     9 orange
                '#715040', '#9E957D', '#595299', '#CB5D7C', '#C6875D',
            #   -10 dark   -9 blue    -8 blue    -7 blue    -6 light
                '#304E68', '#3D6384', '#4D7DA8', '#6D9BC3', '#A7CBE8',
            #   -5 : 0.2   -4 : 0.5   -3 : 0.65  -2 : 0.75  -1 : 0.85
                '#353535', '#7F7F7F', '#A5A5A5', '#BFBFBF', '#D9D9D9'
            )),
        'xaxis': dict(
            title={
                'font': dict(color='#707070', size=13),
                'standoff': 10
            },
            # rangeslider_visible=False,
            anchor='free',
            position=0,
            color='#707070',
            griddash='1px',
            gridwidth=1,
            # gridcolor='#EBEBEB',
            gridcolor='#E1E1E1',
            linecolor='#E1E1E1',
            linewidth=1,
            showticklabels=True,
            ticks='outside',
            ticklen=7,
            tickwidth=1,
            tickcolor='rgba(1,1,1,0)',
            # tickcolor='#E1E1E1',
            # tickfont=dict(color='#757575', size=11),
            tickfont=dict(color='#505050', size=11),
            showgrid=False,
            showline=True,
            showspikes=False,
            spikedash='3px',
            spikecolor='#AAAAAA',
            spikesnap="cursor",
            spikemode="across",
            spikethickness=-2,
            zeroline=False,
            tickangle=0
            ),
        'yaxis': dict(
            title={
                'font': dict(color='#707070', size=13),
                'standoff': 10
            },
            anchor='free',
            color='#707070',
            griddash='1px',
            gridwidth=1,
            # gridcolor='#EBEBEB',
            gridcolor='#E1E1E1',
            linecolor='#E1E1E1',
            linewidth=1,
            showticklabels=True,
            ticks='outside',
            ticklen=7,
            tickwidth=1,
            tickcolor='rgba(1,1,1,0)',
            # tickcolor='#E1E1E1',
            # tickfont=dict(color='#757575', size=11),
            tickfont=dict(color='#505050', size=11),
            position=0,
            showgrid=False,
            showline=True,
            showspikes=False,
            spikedash='3px',
            spikecolor='#DADADA',
            spikesnap="cursor",
            spikemode="across",
            spikethickness=-2,
            zeroline=False,
            tickangle=0
            ),
        'legend': dict(
            traceorder='normal',
            title={
                'font': dict(
                    color='#404040',
                    size=14,
                    style='normal',
                    weight='bold')
            },
            font = dict(
                color='#505050',
                size=12,
                style='normal',
                weight='normal'
            ),
            orientation='h',
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0,
            # entrywidth=40,
            itemsizing='trace',
            itemwidth=30,
            tracegroupgap=10,
            indentation=20
        ),
        'hovermode': 'x', # x unified
        'hoverlabel': dict(
            namelength=-1,
            # align='right',
            bgcolor='rgba(255, 255, 255, 0.9)',
            # bgcolor='#FFFFFF',
            bordercolor='#A5A5A5',
            font=dict(
                family=font_family,
                weight='normal',
                color='#404040',
            )),
        'modebar': dict(
            orientation='v',
            bgcolor='rgba(1,1,1,0)',
            color='rgba(1,1,1,0.5)',
            activecolor='rgba(1,1,1,0.7)',
        ),
        # 'paper_bgcolor': 'rgba(1,1,1,0)', # прозрачный фон
        # 'plot_bgcolor': 'rgba(1,1,1,0)', # прозрачный фон
        'paper_bgcolor': '#FFFFFF', # FAFFFB
        'plot_bgcolor': '#FFFFFF',  
    },
    data = {
        # Each graph object must be in a tuple or list for each trace
        'bar': [
            go.Bar(
                # texttemplate ='%{value:$.2s}',
                textposition='none',
                textfont= {
                    'size': 10,
                    'color': '#FFFFFF'
                    })
            ],
        'scatter': [
            go.Scatter(
                line={
                    'width': 1.5},
                marker={
                    'size': 3}
            )
        ]
    }
)

config = dict(
    scrollZoom=False,
    showLink=False,
    displaylogo=False,
    # displayModeBar=False,
    locale='ru',
    responsive=True,
    toImageButtonOptions = {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'chart',
        # 'width': 1280,
        # 'height': 810,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
        },
    modeBarButtons = [
        [
            'resetScale2d',
            'toImage',
            ]
      ],
    )
config_wo_modebar = config.copy()
config_wo_modebar['displayModeBar'] = False

pio.templates.default = 'primetheme'
palette = pio.templates['primetheme'].layout.colorway
palette_saturated = saturate_palette(palette, 0.85, 'HEX')


# // --- LOAD DATA --- //

path_current = os.getcwd()
path_files = path_current + '/files'
path_assets = path_current + '/docs/'

economics_data = loadit('economics_data', path=path_files, create_empty_dict=True)
cpi_real_time_groups_mean = economics_data['cpi_real_time_groups_mean'].copy()
cpi_kipc_primary_perc = economics_data['cpi_kipc_primary_perc'].copy()
prices_food_growth = economics_data['prices_food_growth'].copy()
price_structure = economics_data['price_structure'].copy()


# // --- PREPARE DATA --- //

current_year = cpi_real_time_groups_mean.index[-1].year
previous_year = current_year - 1
current_month = dt.datetime.strftime(cpi_kipc_primary_perc.index[-1], '%B')
current_month_rus = months_translate(current_month, kind='eng-rus')
current_month_rus_year = months_translate(current_month, kind='eng-rus', add_year=current_year)
current_month_rus_sklon = months_translate(current_month, kind='eng-rus', sklon='pred')
current_month_rus_sklon_year = \
    months_translate(current_month, kind='eng-rus', add_year=current_year, sklon='pred')

xtickaliases_w_year = pl_tickaliases(arange(2021, 2024, True))
xtickaliases_wo_year = pl_tickaliases(arange(2021, 2024, True), add_year_jan=False)
xtickaliases_short_wo_year = pl_tickaliases(arange(2021, 2024, True), add_year_jan=False, fullname=False)
xtickaliases_full_w_year = pl_tickaliases(arange(2021, 2024, True), fullname=True)
xtickaliases_full_wo_year = pl_tickaliases(arange(2021, 2024, True), add_year_jan=False, fullname=True)

xticktext_full_all = pl_xticklabels_rus(arange(2021, 2024, True), add_year='all', fullname=True)

cpi_real_time = pd.DataFrame(
    data=np.column_stack((
        cpi_real_time_groups_mean.iloc[-1:, :].mean(axis=0).round(2),
        cpi_real_time_groups_mean.iloc[-2:, :].mean(axis=0).round(2),
        cpi_real_time_groups_mean.iloc[-4:, :].mean(axis=0).round(2),
        cpi_real_time_groups_mean.loc[str(previous_year)].mean(axis=0).round(2),
    )),
    index=cpi_real_time_groups_mean.iloc[-1:, :].mean(axis=0).index,
    columns=['Текущий', '2 недели', '1 месяц', 'Предыдущий год']
)
cpi_real_time['Предыдущий год был'] = [str(previous_year)]*len(cpi_real_time)
cpi_real_time = cpi_real_time.sort_values('Текущий')

xticks_real_time_min = cpi_real_time['Текущий'].iloc[0]
xticks_real_time_max = cpi_real_time['Текущий'].mean().round(2)

cpi_real_time_index_all = cpi_real_time['Текущий'].index.tolist().index('Среднее значение')
cpi_real_time_colors = [alpha_color(palette[2], 1, 'HEX')]*len(cpi_real_time)
cpi_real_time_colors[cpi_real_time_index_all] = saturate_color(palette[1], 1, 'HEX')

prices_food_growth = prices_food_growth*100-100
# prices_food_growth_smoothed = smoothed(prices_food_growth, datetime_index=True)

cpi_kipc_primary_types = cpi_kipc_primary_perc['Тип'].unique().tolist()
cpi_kipc_primary_perc_period_previous = \
    cpi_kipc_primary_perc[cpi_kipc_primary_perc['Тип'] == cpi_kipc_primary_types[2]].iloc[:, 1:].copy()

cpi_real_value = \
    round(cpi_kipc_primary_perc_period_previous.iloc[-1, :]['Все товары и услуги'], 1)
cpi_target_value = 4.0
cpi_week_target_value = pow(((104 * 100**52)/100), 1/52) - 100
cpi_week_target_value = round(cpi_week_target_value, 3)
cpi_real_color = palette[2]
if cpi_real_value > cpi_target_value:
    cpi_target_color = palette[1]
else:
    cpi_target_color = palette[3]
cpi_month_value = \
    dt.datetime.strftime(cpi_kipc_primary_perc_period_previous.index[-1], '%B %Y')
cpi_month_value = \
    date_translate(cpi_month_value, date_format='%B %Y', kind='eng-rus', sklon='pred')


cpi_linechart_this_year = cpi_kipc_primary_perc_period_previous.loc['2021':, 'Все товары и услуги'].copy()

if len(cpi_linechart_this_year) > 7:
    xtickaliases_cpi_linechart = xtickaliases_w_year
else:
    xtickaliases_cpi_linechart = xtickaliases_full_w_year

window_stop_index_for_rolling = \
    cpi_real_time_groups_mean.loc[cpi_real_time_groups_mean.index < str(previous_year)].iloc[-5].name

cpi_real_time_groups_mean_rolling = \
    cpi_real_time_groups_mean.loc[
        cpi_real_time_groups_mean.index > window_stop_index_for_rolling,
        'Среднее значение'].rolling(5).mean()

cpi_week_rolling_previous = cpi_real_time_groups_mean_rolling.loc[str(previous_year)].round(2)
cpi_week_rolling_current = cpi_real_time_groups_mean_rolling.loc[str(current_year)].round(2)

cpi_real_time_trend_diff = (cpi_week_rolling_current.values
                            - cpi_week_rolling_previous.values[:len(cpi_week_rolling_current)]).round(2)

cpi_real_time_trend_customdata_previous = \
    pl_ticktext_transform_dates_to_rus(cpi_week_rolling_previous.index)
cpi_real_time_trend_customdata_current = \
    pl_ticktext_transform_dates_to_rus(cpi_week_rolling_current.index)

len_cpi_real_time_trend_customdata_current = len(cpi_real_time_trend_customdata_current)

cpi_real_time_trend_customdata_previous[:len_cpi_real_time_trend_customdata_current] = \
    cpi_real_time_trend_customdata_current

# cpi_real_time_trend_diff_colors = \
    # [palette[2] if i > 0 else palette[0] for i in cpi_real_time_trend_diff]


# // --- PREPARE CHARTS --- //

# CPI KPI LINE-CHARTS

fig_cpi_linechart_this_year = go.Figure()

# fig_cpi_linechart_this_year.add_trace(
#     go.Scatter(
#         x=smoothed(cpi_linechart_this_year, datetime_index=True).index,
#         y=smoothed(cpi_linechart_this_year, datetime_index=True).values.ravel(),
#         mode='lines', line=dict(width=3, color=saturate_color(palette[2], 0.8, 'HEX')),
#         hoverinfo='skip', showlegend=False, name=''
#     )
# )
fig_cpi_linechart_this_year.add_trace(
    go.Scatter(
        x=cpi_linechart_this_year.index,
        y=cpi_linechart_this_year,
        mode='lines',
        line=dict(shape='spline', width=3, color=saturate_color(palette[2], 0.8, 'HEX')),
        hoverinfo='skip', showlegend=False, name=''
    )
)
fig_cpi_linechart_this_year.add_trace(
    go.Scatter(
        x=cpi_linechart_this_year.index,
        y=cpi_linechart_this_year,
        mode='markers', marker=dict(color=palette[2], size=7, opacity=1),
        showlegend=False, name='',
        text=xticktext_full_all,
        hovertemplate=
            '%{text}'
            + '<br>Инфляция: <b>%{y}%</b>'
    )
)
pl_hline(
    cpi_target_value, width=2, line_dash='solid', opacity=0.75, color=cpi_target_color,
    showlegend=False, figure=fig_cpi_linechart_this_year
)
fig_cpi_linechart_this_year.update_layout(
    margin=dict(b=50, r=30, l=35, t=10),
    hovermode='x',
    # hovermode='closest',
    xaxis=dict(
        dtick='M3',
        tickformat="%b<br>%Y",
        labelalias=xtickaliases_cpi_linechart,
        showspikes=True,
        showline=True,
        tickcolor='#FFFFFF',
        ticklen=5,
        showgrid=True,
    ),
    yaxis=dict(
        showline=False,
        showgrid=True,
    )
)


# CPI REAL-TIME CHARTS


xticks_real_time = arange(len(cpi_real_time))
xticklabels_real_time = cpi_real_time.index.tolist()
xticktext_real_time = xticklabels_real_time.copy()
# xticklabels_real_time_break = ['Молочные продукты']
# xticktext_real_time = \
#     [add_break_after_index(i) if i in xticklabels_real_time_break
#      else i for i in xticklabels_real_time]
bar_width_real_time = 0.5

# cpi real-time main plot
fig_cpi_real_time_groups = go.Figure()

fig_cpi_real_time_groups.add_trace(
    go.Bar(
        x=cpi_real_time['Текущий'],
        y=xticks_real_time,
        width=bar_width_real_time,
        orientation='h',
        marker_color=cpi_real_time_colors,
        showlegend=True,
        name='Текущая неделя',
        text=xticklabels_real_time,
        customdata=cpi_real_time,
        hovertemplate=
            '<b>%{text}</b>'
            + '<br>Значение на текущей неделе: <b>%{customdata[0]}%</b>'
            + '<br>Среднее за две предыдущие недели: %{customdata[1]}%'
            + '<br>Среднее за предыдущий месяц: %{customdata[2]}%'
            + '<br>Среднее за %{customdata[4]} год: %{customdata[3]}%'
            + '<extra></extra>'
    )
)
fig_cpi_real_time_groups.add_trace(
    go.Bar(
        x=cpi_real_time['2 недели'],
        y=xticks_real_time,
        width=bar_width_real_time,
        orientation='h',
        marker_color=alpha_color(palette[0], 0.5, 'HEX'), showlegend=True,
        name='Две предыдущие недели',
        text=xticklabels_real_time,
        customdata=cpi_real_time,
        hovertemplate=
            '<b>%{text}</b>'
            + '<br>Значение на текущей неделе: <b>%{customdata[0]}%</b>'
            + '<br>Среднее за две предыдущие недели: %{customdata[1]}%'
            + '<br>Среднее за предыдущий месяц: %{customdata[2]}%'
            + '<br>Среднее за %{customdata[4]} год: %{customdata[3]}%'
            + '<extra></extra>'
    )
)
fig_cpi_real_time_groups.add_trace(
    go.Bar(
        x=cpi_real_time['1 месяц'],
        y=xticks_real_time,
        width=bar_width_real_time,
        orientation='h',
        marker_color=alpha_color(palette[0], 0.25, 'HEX'), showlegend=True,
        name='Предыдущий месяц',
        text=xticklabels_real_time,
        customdata=cpi_real_time,
        hovertemplate=
            '<b>%{text}</b>'
            + '<br>Значение на текущей неделе: <b>%{customdata[0]}%</b>'
            + '<br>Среднее за две предыдущие недели: %{customdata[1]}%'
            + '<br>Среднее за предыдущий месяц: %{customdata[2]}%'
            + '<br>Среднее за %{customdata[4]} год: %{customdata[3]}%'
            + '<extra></extra>'
    )
)
fig_cpi_real_time_groups.add_trace(
    go.Scatter(
        mode='markers',
        x=cpi_real_time['Предыдущий год'],
        y=xticks_real_time,
        marker=dict(color=alpha_color('#000000', 0.75, 'HEX'),  size=7, symbol='diamond-tall'),
        showlegend=True, hoverinfo='skip',
        name=f'Среднее за {previous_year} год'
    )
)
fig_cpi_real_time_groups.add_shape(
    x0=xticks_real_time_min,
    x1=xticks_real_time_max,
    y0=len(cpi_real_time),
    y1=len(cpi_real_time),
    type='line', line_color='#808080', line_width=1
)
fig_cpi_real_time_groups.update_layout(
    margin=dict(t=35, l=145, r=0, b=65),
    xaxis=dict(
        showgrid=False,
        showspikes=False,
        showline=False,
        anchor='free',
        position=1,
        side='top',
        tickcolor='#808080',
        ticklen=6,
        tickwidth=1,
        tickvals=[xticks_real_time_min, xticks_real_time_max]
    ),
    yaxis=dict(
        showline=True,
        tickformat='%B %d',
        tickvals=xticks_real_time,
        ticktext=xticktext_real_time,
        ticklen=10,
    ),
    barmode='relative',
    hovermode='closest',
    hoverlabel_align = 'left',
    legend=dict(
        x=0,
        y=-0.15,
        entrywidth=155,
    ),
    modebar=dict(
        orientation='v'
    )
)

# cpi real-time trend rolling
fig_cpi_real_time_trend = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,  row_width=[0.5, 1])

fig_cpi_real_time_trend.add_trace(
    go.Scatter(
        x=cpi_week_rolling_previous.index,
        y=cpi_week_rolling_previous,
        line=dict(width=2, color=palette[0], shape='spline'),
        name=str(previous_year),
        hovertemplate='%{y}%'
    ), row=1, col=1
)
fig_cpi_real_time_trend.add_trace(
    go.Scatter(
        x=cpi_week_rolling_previous.index,
        y=cpi_week_rolling_current,
        line=dict(width=2, color=palette[2], shape='spline'),
        name=str(current_year),
        hovertemplate='%{y}%'
    ), row=1, col=1
)
fig_cpi_real_time_trend.add_trace(
    go.Scatter(
        x=cpi_week_rolling_previous.index,
        y=[cpi_week_target_value]*len(cpi_week_rolling_previous.index),
        name='Цель ЦБ',
        mode='lines',
        line_width=2,
        line_color=palette[1],
        opacity=0.75,
        showlegend=True,
        hovertemplate='%{y}%'
        # hoverinfo='skip',
    ),  row=1, col=1
)
fig_cpi_real_time_trend.add_shape(
    x0=cpi_week_rolling_previous.index[0],
    x1=cpi_week_rolling_previous.index[-1],
    y0=0,
    y1=0,
    type='line',
    line_color='#808080',
    line_width=1,
    line_dash='5px',
    row=1, col=1
)

fig_cpi_real_time_trend.add_trace(
    go.Bar(
        x=cpi_week_rolling_previous.index,
        y=cpi_real_time_trend_diff,
        name='Разность 2024 и 2023',
        showlegend=True,
        hoverinfo='skip',
        marker_color=palette[2]
    ), row=2, col=1
)
fig_cpi_real_time_trend.add_annotation(
    text='1). Используется средняя инфляция по всем группам товаров за 5 недель.',
    x=-0.005, y=-0.275,
    xref="paper", yref="paper",
    showarrow=False,
)
fig_cpi_real_time_trend.update_layout(
    margin=dict(t=10, b=60, r=10, l=50),
    xaxis1 = dict(
        ticks='',
        dtick='M1',
        tickcolor='#E1E1E1',
        ticklen=5,
        tickvals=cpi_week_rolling_previous.index,
        ticktext=cpi_real_time_trend_customdata_previous,
    ),
    xaxis2=dict(
        ticks='outside',
        tickcolor='#E1E1E1',
        ticklen=5,
        dtick='M1',
        tickformat="%b<br>%Y",
        labelalias=xtickaliases_short_wo_year,
    ),
    yaxis2=dict(
        tickfont=dict(size=11),
        dtick=0.1
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='rgba(255, 255, 255, 0.9)'
    ),
    bargap=0.35,
    modebar=dict(
        orientation='v'
    ),
    legend=dict(
        y=1.04,
        yanchor='bottom',
        xanchor='left',
    )
)


# CPI KIPC
# cpi_kipc_smoothed = smoothed(cpi_kipc, datetime_index=True)

cpi_kipc_dropdown_values = [
    'К декабрю предыдущего года',
    'К предыдущему месяцу',
    'К соотв. периоду предыдущего года',
    'С начала года к соотв. периоду предыдущего года'
]

cpi_kipc_num = \
    cpi_kipc_primary_perc[cpi_kipc_primary_perc['Тип'] == 'К соответствующему периоду предыдущего года'].iloc[:, 1:].copy()

xticks_kipc = arange(len(cpi_kipc_num.columns))
xticklabels_kipc = [
    'Все<br>товары и<br>услуги', 'Продукты<br>питания', 'Алкоголь<br>и табак', 'Одежда', 'ЖКХ',
    'Товары<br>для дома', 'Медицина', 'Транспорт', 'Сотовая<br>связь', 'Культура',
    'Образова-<br>ние', 'Кафе и<br>рестораны'
]
xticktext_kipc = [
    'Все товары и услуги', 'Продукты питания', 'Алкоголь и табак', 'Одежда', 'ЖКХ',
    'Товары для дома', 'Медицина', 'Транспорт', 'Сотовая связь', 'Культура',
    'Образование', 'Кафе и рестораны'
]
current_month_kipc = dt.datetime.strftime(cpi_kipc_num.iloc[-1, :].name, '%B')
current_month_kipc = (months_translate(current_month_kipc, 'eng-rus')
                 + ' '
                 + dt.datetime.strftime(cpi_kipc_num.iloc[-1, :].name, '%Y'))
xticklength_kipc = len(xticks_kipc)
customdata_kipc = list([current_month_kipc]) * xticklength_kipc
customdata_kipc = [[i] for i in customdata_kipc]

# main plot
fig_cpi_kipc = go.Figure()

fig_cpi_kipc.add_trace(
    go.Bar(
        x=xticks_kipc,
        y=cpi_kipc_num.iloc[-1:, :].values.ravel(),
        width=0.5,
        marker_color=saturate_color(palette[2], 0.75, 'HEX'), showlegend=False, name='',
        customdata=customdata_kipc,
        hovertemplate=
            '<b>%{text}</b>'
            + '<br>%{customdata[0]}'
            + '<br>Изменение цены: %{y}%',
            # + "<br><a href='%{customdata[1]}'>Подробная информация</a><extra></extra>",
        text = xticktext_kipc,
        hoverinfo = 'name+z',
    )
)
pl_hline(
    cpi_target_value, width=2, line_dash='solid', opacity=0.75, color=cpi_target_color,
    showlegend=True,
    name='Цель ЦБ по инфляции',
    figure=fig_cpi_kipc
)
fig_cpi_kipc.update_layout(
    # width=850,
    # height=300,
    margin=dict(b=70, t=8),
    xaxis=dict(
        tickvals=xticks_kipc,
        ticktext=xticklabels_kipc,
        showline=False,
        showspikes=False,
        showgrid=False,
    ),
    yaxis = dict(
        showline=False,
        showgrid=True,
        ticklen=10,
    ),
    hoverlabel_align = 'left',
    # hovermode= 'x unified',
    # hovermode= 'x',
    hovermode='closest'
)

# time series
fig_cpi_kipc_ts = go.Figure()

fig_cpi_kipc_ts.add_trace(
    go.Scatter(
        x=smoothed(cpi_kipc_num['Все товары и услуги'], n=1000, datetime_index=True).index,
        y=smoothed(cpi_kipc_num['Все товары и услуги'], n=1000, datetime_index=True).values.ravel(),
        mode='lines', line=dict(width=3, color=saturate_color(palette[2], 0.8, 'HEX')),
        hoverinfo='skip', showlegend=False, name=''
    )
)
fig_cpi_kipc_ts.add_trace(
    go.Scatter(
        x=cpi_kipc_num['Все товары и услуги'].index,
        y=cpi_kipc_num['Все товары и услуги'],
        mode='markers', marker=dict(opacity=0),
        showlegend=False, name='',
        text=['Все товары и услуги']*len(cpi_kipc_num),
        hovertemplate=
            '<b>%{text}</b>'
            + '<br>%{x}'
            + '<br>Изменение цены: %{y}%'
    )
)


# PRICES GROWTH


# prices_food_growth_dropdown_values = [
#     'Хлеб', 'Говядина', 'Свинина', 'Молочные продукты', 'Овощи'
# ]
prices_food_growth_products_dict = {
    'Хлеб': [
        'хлеб и булочные изделия из пшеничной муки различных сортов, кг',
        'пшеница'
    ],
    'Говядина': [
        'говядина (кроме бескостного мяса), кг', 
        'крупный рогатый скот'
    ],
    'Свинина': [
        'свинина (кроме бескостного мяса), кг',
        'свиньи'
    ],
    'Курица': [
        'куры охлажденные и мороженые, кг',
        'птица сельскохозяйственная живая'
    ],
    'Яйца куриные': [
        'яйца куриные, 10 шт.',
        'яйца куриные в скорлупе свежие'
    ],
    'Молочные продукты': [
        'масло сливочное, кг', 'сыры твердые, полутвердые и мягкие, кг', 'сметана, кг',
        'йогурт, кг', 'творог, кг', 'молоко сырое крупного рогатого скота',
        'молоко питьевое цельное пастеризованное 2,5-3,2% жирности, л'
    ],
    'Овощи': [
        'картофель, кг', 'огурцы свежие, кг', 'помидоры свежие, кг', 'лук репчатый, кг',
        'морковь, кг', 'овощи'
    ]
}
prices_food_growth_plots_dict = {
    'хлеб и булочные изделия из пшеничной муки различных сортов, кг': 
        ['Цена хлеба', palette[6], 'solid'],
    'пшеница':
        ['Цена пшеницы', palette[0], '5px'],
    'говядина (кроме бескостного мяса), кг':
        ['Цена говядины', palette[1], 'solid'],
    'крупный рогатый скот':
        ['Цена КРС', palette[0], '5px'],
    'свинина (кроме бескостного мяса), кг':
        ['Цена свинины', palette[8], 'solid'],
    'свиньи': 
        ['Цена свиней у производителей', palette[0], '5px'],
    'куры охлажденные и мороженые, кг':
        ['Цена курицы', palette[7], 'solid'],
    'птица сельскохозяйственная живая':
        ['Цена птицы сельскохозяйственной', palette[0], '5px'],
    'яйца куриные, 10 шт.':
        ['Цены для потребителей', palette[7], 'solid'],
    'яйца куриные в скорлупе свежие':
        ['Цены производителей', palette[0], '5px'],
    'масло сливочное, кг':
        ['Масло сливочное', palette_saturated[1], 'solid'],
    'сыры твердые, полутвердые и мягкие, кг':
        ['Сыр', palette_saturated[2], 'solid'],
    'сметана, кг':
        ['Сметана', palette_saturated[3], 'solid'],
    'йогурт, кг':
        ['Йогурт', palette_saturated[4], 'solid'],
    'творог, кг':
        ['Творог', palette_saturated[7], 'solid'],
    'молоко сырое крупного рогатого скота':
        ['Сырое молоко', palette[0], '5px'],
    'молоко питьевое цельное пастеризованное 2,5-3,2% жирности, л':
        ['Пастеризованное молоко', alpha_color(palette[0], 0.75, return_type='HEX'), 'solid'],
    'картофель, кг':
        ['Картофель', palette_saturated[5], 'solid'],
    'огурцы свежие, кг':
        ['Огурцы', palette_saturated[3], 'solid'],
    'помидоры свежие, кг':
        ['Помидоры', palette_saturated[1], 'solid'],
    'лук репчатый, кг':
        ['Лук', palette_saturated[2], 'solid'],
    'морковь, кг':
        ['Морковь', palette_saturated[9], 'solid'],
    'овощи':
        ['Овощи (средняя цена производителей)', palette[0], '5px']
}
# prices_food_growth_properties_dict = {
#     'Хлеб': [0.1, None],
#     'Говядина': [0.1, None],
#     'Свинина': [0.1, None],
#     'Курица': [0.1, None],
#     'Куриные яйца': [0.2, None],
#     'Молочные продукты': [0.1, 90],
#     'Овощи': [0.5, 90]
# }
prices_food_growth_properties_dict = {
    'Хлеб': [10, None, 2],
    'Говядина': [10, None, 2],
    'Свинина': [5, None, 2],
    'Курица': [10, None, 2],
    'Яйца куриные': [20, None, 2],
    'Молочные продукты': [10, 100, 1.5],
    'Овощи': [30, 90, 1.5]
}


fig_prices_food_growth = go.Figure()
fig_price_structure = go.Figure()

# // --- PAGE FUNCTIONS --- //

def drawFigure(figure, config):
    result = html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(figure=figure, config=config) 
            ])
        ),  
    ])
    return result


def drawFigure(figure, config, width):

    result = html.Div([
        dcc.Graph(figure=figure, config=config)
    ], style={'width': width})

    return result

# // --- PAGE CONTENT --- //

separator_icon = html.Div([
    html.Div([html.Img(src='assets/favicon.png', style={'width':'1.35vw'})], className='separator')])

inflation_target = html.Div([
    html.H5('Цель ЦБ по инфляции', className='inflation-kpi-dash-title'),
    html.P(
        html.P(f'{cpi_target_value} %'), className='inflation-kpi-dash-content',
        style={'color': f'{cpi_target_color}'})
], className='inflation-kpi-dash-container')

inflation_real = html.Div([
    html.H5(f'Инфляция в {cpi_month_value}', className='inflation-kpi-dash-title'),
    html.P(
        html.P(f'{cpi_real_value} %'), className='inflation-kpi-dash-content',
        style={'color': f'{cpi_real_color}'})
], className='inflation-kpi-dash-container')

inflation_forecasts = html.Div([
    html.Div([
        html.Div([
            html.H5('Центральный банк', className='forecasts-title'),
            html.P('4.3% - 4.8%', className='forecasts-content', style={
                # 'color':saturate_color(palette[3], 1, 'HEX'),
                'color': '#808080',
                'font-weight':'500'
            })
        ], className='forecasts-container'),
        html.Div([
            html.H5('Минфин', className='forecasts-title'),
            html.P('5.1%', className='forecasts-content', style={
                'color':saturate_color(palette[-10], 1, 'HEX'),
                # 'color': '#6063A3'
            })
        ], className='forecasts-container', style={'width':'18%'}),
        html.Div([
            html.H5('Минэк', className='forecasts-title'),
            html.P('5.1%', className='forecasts-content', style={
                'color':saturate_color(palette[-8], 1, 'HEX'),
            })
        ], className='forecasts-container', style={'width':'16%', 'margin-left':'1vw'}),
        html.Div([
            html.H5('Всемирный банк', className='forecasts-title'),
            html.P('6.9%', className='forecasts-content', style={
                # 'color':saturate_color(palette[7], 1, 'HEX'),
                'color':saturate_color(palette[3], 1, 'HEX'),
                # 'color': '#5B67BD'
            })
        ], className='forecasts-container'),
    ], className='forecasts-main-container') 
], style={'height':'100%'})

header_big = html.Div([
    html.Div([], className='hr-header-big-top'),
    html.Div([
        html.Div([html.A(['Главная'], href='/', className='main')], className='header-big-button-active'),
        html.Div([html.A(['Инфляция и цены'], href='/inflation-and-prices', className='inflation-hover')], className='header-big-button'),
        html.Div([html.A(['Бюджет'], href='/budget', className='budget-hover')], className='header-big-button'),
        # html.Div([html.A(['Промышленность'], href='/page-1'),], className='header-big-button')
    ], className='header-big-container'),
    html.Div([], className='hr-header-big-bottom'),
    html.Div([separator_icon], style={'margin-bottom':'1vh'})
])
header_big_inflation = html.Div([
    html.Div([], className='hr-header-big-top'),
    html.Div([
        html.Div([html.A(['Главная'], href='/')], className='header-big-button'),
        html.Div([html.A(['Инфляция и цены'], href='/inflation-and-prices', className='inflation')], className='header-big-button-active'),
        html.Div([html.A(['Бюджет'], href='/budget', className='budget-hover')], className='header-big-button'),
        # html.Div([html.A(['Промышленность'], href='/page-1'),], className='header-big-button')
    ], className='header-big-container'),
    html.Div([], className='hr-header-big-bottom'),
    # separator_icon
    html.Div([separator_icon], style={'margin-bottom':'1vh'})
])
header_big_budget = html.Div([
    html.Div([], className='hr-header-big-top'),
    html.Div([
        html.Div([html.A(['Главная'], href='/')], className='header-big-button'),
        html.Div([html.A(['Инфляция и цены'], href='/inflation-and-prices', className='inflation-hover')], className='header-big-button'),
        html.Div([html.A(['Бюджет'], href='/budget', className='budget')], className='header-big-button-active'),
        # html.Div([html.A(['Промышленность'], href='/page-1'),], className='header-big-button')
    ], className='header-big-container'),
    html.Div([], className='hr-header-big-bottom'),
    # separator_icon
    html.Div([separator_icon], style={'margin-bottom':'1vh'})
])

header_small_inflation = html.Div([
    html.Div([
        html.Div([
            html.Div(
                html.A('Инфляция', href='/inflation-and-prices'), className='header-small-button-active'),
            html.Div(
                html.A('Цены', href='/prices'), className='header-small-button'),
        #     html.Div(
        #         html.A('Инфляция в 1990-х', href='/inflation-90'), className='header-small-button', style={'width':'12vw'}),
        ], style={'width':'99vw', 'margin-left':'1vw', 'display':'flex', 'align-items': 'center'})
    ], className='header-small-container')
])
header_small_prices = html.Div([
    html.Div([
        html.Div([
            html.Div(
                html.A('Инфляция', href='/inflation-and-prices'), className='header-small-button'),
            html.Div(
                html.A('Цены', href='/prices'), className='header-small-button-active'),
        #     html.Div(
        #         html.A('Инфляция в 1990-х', href='/inflation-90'), className='header-small-button', style={'width':'12vw'}),
        ], style={'margin-left':'1vw', 'display':'flex', 'align-items': 'center'})
    ], className='header-small-container')
])
header_small_inflation_90 = html.Div([
    html.Div([
        html.Div([
            html.Div(
                html.A('Инфляция', href='/inflation-and-prices'), className='header-small-button'),
            html.Div(
                html.A('Цены', href='/prices'), className='header-small-button'),
        #     html.Div(
        #         html.A('Инфляция в 1990-х', href='/inflation-90'), className='header-small-button-active', style={'width':'12vw'}),
        ], style={'margin-left':'1vw', 'display':'flex', 'align-items': 'center'})
    ], className='header-small-container')
])

content_under_development = html.Div([
    html.Div(
        [html.Img(src='assets/under-construction.png', style={'width':'5vw'})],
        style={'margin-top':'1vh', 'display':'flex', 'justify-content':'center'}),
    html.Div(['Страница в разработке'], style={'margin-top':'1vh', 'font-size':'1em', 'text-align': 'center'}),
], style={'display': 'inline'})

page_start = html.Div([
    # header
    header_big,
    html.Div([
        # content
        content_under_development
    ], className='page-container')
])

page_inflation = html.Div([
    # header
    header_big_inflation,
    header_small_inflation,
    # content
    html.Div([
        # first row
        html.Div([
            # left title
            html.Div(
                html.H4('Ключевые показатели'),
                style={
                    'float':'left',
                    'width':'44vw',
                }),
            # right title
            html.Div(
                    html.H4('Недельные данные'),
                    style={
                        'float':'left',
                        'width':'52vw',
                        'margin-left':'1vw'
                    }),
            # left big dash
            html.Div([
                # left little dash
                html.Div([
                    html.Div(
                        inflation_target,
                        style={
                            'float':'left',
                            'width':'50%',
                        }),
                    # right little dash
                    html.Div(
                        inflation_real,
                        style={
                            'float':'left',
                            'width':'50%',
                        }),
                ], style={
                    'height':'20%',
                    'display':'flex',
                    'align-items':'center',
                    'padding':'0 5vw 0 5vw',
                }),
                html.Div(className='hr-grey', style={'width':'96%', 'margin':'auto'}),
                html.H6(['Прогнозы на год'], style={
                    'height':'7%',
                    'padding':'0 0 0 1vw',
                    'text-align':'left',
                    # 'border':'1px solid red',
                    }),
                html.Div(
                    inflation_forecasts,
                    style={
                        'width':'100%',
                        'height':'16%',
                    }),
                html.Div(className='hr-grey', style={'width':'96%', 'margin':'auto'}),
                # line-chart title
                html.H6(['Динамика с 2021 года'], style={
                    'height':'7%',
                    'padding':'0 0 0 1vw',
                    'text-align':'left',
                }),
                # bottom little line-chart
                html.Div([
                    dcc.Graph(
                        className='graph-figure',
                        figure=fig_cpi_linechart_this_year, config=config)
                ], style={
                    # 'float':'top',
                    'height':'49%',
                    # 'display': 'inline-block'
                }),
            ], className='content-container', style={'width':'44vw', 'height':'55vh'}),
            # right big dash
            html.Div([
                html.H6(['Основные категории товаров'], style={
                    'height':'3vh',
                    'padding':'0 0 0 1vw',
                    'text-align':'left',
                    'align-items':'end',
                }),
                html.Div([
                    dcc.Graph(
                    className='graph-figure',
                    figure=fig_cpi_real_time_groups, config=config)
                ], style={
                    'height':'52vh',
                })
            ], className='content-container', style={'width':'53vw', 'height':'55vh', 'margin-left':'1vw'})
        ], style={'width':'100%'}),
        # second row
        html.Div([
            # bottom line-chart title
            html.H4(f'Инфляция в {current_month_rus_sklon_year}', style={'width':'58vw', 'float':'left'}),
            html.H4('5-недельная инфляция', style={'width':'39vw', 'margin-left':'1vw', 'float':'left'}),
            # bottom big line-chart 
            html.Div([
                dcc.Graph(
                    className='graph-figure',
                    figure=fig_cpi_kipc, config=config)
            ], className='content-container', style={'width':'56vw', 'height':'35vh'}),
            html.Div([
                dcc.Graph(
                    className='graph-figure',
                    figure=fig_cpi_real_time_trend, config=config)
            ], className='content-container', style={'width':'41vw', 'height':'35vh', 'margin-left':'1vw'})
        ], style={
            # 'float':'left',
            'width':'100%'})
    ], className='page-container')
])

def prices_radioitems_generate_items(values):
    result = []
    for item in values:
        result.append({'label': item, 'value': item})
    return result
    
prices_radioitems = html.Div([
    dbc.Label('Категория', style={
        'font-weight':'500',
        # 'text-align':'center'
    }),
    # dbc.RadioItems(
    #     options=prices_radioitems_generate_items(prices_food_growth_products_dict.keys()),
    #     value='Хлеб',
    #     id='prices-radioitems-input'),
    dcc.RadioItems(
        options=prices_radioitems_generate_items(prices_food_growth_products_dict.keys()),
        value='Хлеб',
        id='prices-radioitems-input',
        className='prices-radioitems-input'
    )
    ]
)

page_prices = html.Div([
    # header
    header_big_inflation,
    header_small_prices,
    # content
    html.Div([
        # food prices
        html.Div([
            # title
            html.Div(html.H4('Цены на продукты питания'),
            style={'float':'left', 'width':'75vw', 'margin':'0 0 0 1vw'}),
            # sub-title
            html.Div(
                html.H5('Процентное изменение цен по сравнению с Январем 2021',
                style={'text-align':'left', 'margin-top':'0vh'}),
                style={'float':'left', 'width':'80vw', 'margin':'0 0 0 1vw'}),
            # chart, radioitems, prices structure
            html.Div([
                # chart and radioitems container
                html.Div([
                    # chart container
                    html.Div([
                        dcc.Graph(
                                id='figure-prices-food-growth', className='graph-figure',
                                figure=fig_prices_food_growth, config=config)
                    ], style={'width':'64vw', 'height':'100%', 'float':'left'}),
                    # border
                    html.Div([], className='vr-grey', style={'height':'36vh', 'margin':'0 2vw 0 1vw', 'float':'left'}),
                    # raioitems container
                    html.Div([
                        prices_radioitems
                    ], style={
                        'width':'17vw',
                        'height':'100%',
                        'float':'left',
                        'margin':'auto 0',
                        'display':'flex',
                        'align-items': 'center',
                        'font-size':'1em',
                      # 'justify-content': 'center',
                      # 'border':'1px solid green'
                         }),
                ], style={'width':'100%', 'height':'40vh', 'display':'flex'}),
                # prices structure container
                html.Div([
                    # top border
                    # title
                    html.H5('Структура цены', style={
                        'font-size':'0.9em',
                        'text-align':'left',
                        'margin':'0 0 0 3.5vw'
                    }),
                    # prices structures
                    html.Div([
                        dcc.Graph(
                            id='prices-structure', className='graph-figure',
                            figure=fig_price_structure, config=config_wo_modebar)
                        ], style={
                        'width':'100%',
                        'height':'28vh',
                        'float':'left',
                        'padding-left':'2vw',
                        'padding-right':'2vw'
                    }),
                    ], style={'width':'63vw', 'height':'30vh', 'display':'inline-block'}),
            ], className='content-container', style={'width':'85vw', 'height':'72vh', 'display':'inline-block'}),
        ]),  
    ], className='page-container')
])

# page_inflation_90 = html.Div([
#     # header
#     header_big_inflation,
#     header_small_inflation_90,
#     content_under_development
# ])
page_budget = html.Div([
    header_big_budget,
    content_under_development
])



@callback(
    Output("page-content", "children"),
    Input("url", "pathname"))
def rus_economy_function(pathname):
    if pathname == "/":
        return page_start
    elif pathname == '/inflation-and-prices':
        return page_inflation
    elif pathname == "/prices":
        return page_prices
    elif pathname == "/inflation-90":
        return page_inflation_90
    elif pathname == "/budget":
        return page_budget

@callback(
    Output('figure-prices-food-growth', 'figure'),
    Output('prices-structure', 'figure'),
    Input('prices-radioitems-input', 'value'),
    Input('figure-prices-food-growth', 'figure'),
    Input('prices-structure', 'figure'),
)
def update_prices_gowth_plot(value, figure1, figure2):
    '''
    value - группа товаров ('Хлеб', 'Овощи')
    '''
    # if value is None:
    #     fig_prices_food_growth = figure1
    #     fig_price_structure = figure2

    # else:
    # PRICES GROWTH
    # prices_food_growth = prices_food_growth_smoothed[prices_food_growth_products_dict[value]].copy()
    df = prices_food_growth[prices_food_growth_products_dict[value]].copy()
    df_len = len(df)
    # prices_food_growth = prices_food_growth[prices_food_growth_products_dict[value]].copy()
    
    # if isinstance(prices_food_growth_3, pd.Series):
    #     prices_food_growth_3 = prices_food_growth_3.to_frame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    fig_prices_food_growth = go.Figure()

    # dates
    fig_prices_food_growth.add_trace(
        go.Scatter(
            x=df.index,
            y=[0]*df_len,
            showlegend=False,
            mode='markers',
            marker_color=palette[0],
            marker_size=0,
            opacity=0,
            text=xticktext_full_all,
            hoverinfo='x',
            hovertemplate=
                '<b>%{text}</b>'
                '<extra></extra>'
        )
    )
    # charts
    for col in df.columns:
        fig_prices_food_growth.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            line_shape='spline',
            line_width=prices_food_growth_properties_dict[value][2],
            line_color=prices_food_growth_plots_dict[col][1],
            line_dash=prices_food_growth_plots_dict[col][2],
            name=prices_food_growth_plots_dict[col][0],
            text=[prices_food_growth_plots_dict[col][0]]*len(df.index),
            hovertemplate=
                    '%{text}: <b>%{y}</b>'
                    '<extra></extra>'
        )
    )
    # zero-line
    fig_prices_food_growth.add_hline(
        y=0,
        opacity=1,
        line_color='#000000',
        line_width=0.45,
        line_dash='2px'
                
    )
    # fig_prices_food_growth.add_trace(
    #     go.Scatter(
    #         x=[df.index[0], df.index[-1]],
    #         y=[0]*df_len,
    #         line_color='#707070',
    #         hoverinfo='skip',
    #         showlegend=False,
    #     )
    # )

    fig_prices_food_growth.update_layout(
        margin=dict(b=55, r=10, l=60),
        hovermode='x unified',
        xaxis=dict(
            dtick='M3',
            tickformat="%b<br>%Y",
            labelalias=xtickaliases_w_year,
            hoverformat='%',
            showspikes=True,
        ),
        yaxis=dict(
            dtick=prices_food_growth_properties_dict[value][0],
            ticksuffix=' %',
            showticksuffix = 'all',
            showgrid=True
        ),
        legend=dict(
            entrywidth=prices_food_growth_properties_dict[value][1],
            y=1.05,
            font=dict(
                size=14,
            ),
        ),
        modebar=dict(
            orientation='h'
        )
    )

    # # PRICES STRUCTURE
    if value == 'Овощи':
        fig_price_structure = go.Figure()
        fig_price_structure.update_layout(
            margin=dict(l=75),
            xaxis=dict(
                visible=False
                ),
            yaxis=dict(
                visible=False
                ),
            annotations=[
                dict(
                    text= "Росстат не предоставляет информацию по данной категории.",
                    xref= "paper", yref= "paper",
                    x=-0.077, y=1.05,
                    showarrow= False,
                    font= {"size": 15}
                    )
                ]
            )

    else:
        if value == 'Молочные продукты':
            value = 'Молоко пастеризованное'
    
        fig_price_structure = go.Figure()
    
        fig_price_structure.add_trace(
            go.Bar(
                y=[0]*len(price_structure[value]),
                x=price_structure[value],
                orientation='h',
                width=0.25,
                marker_color=palette_saturated,
                marker_line=dict(width=1, color='#FFFFFF'),
                showlegend=False,
                text=price_structure.index,
                hovertemplate=
                    '<b>%{text}</b>'
                    '<extra></extra>'
            )
        )
        
        pl_legend(
            markers=['p']*len(price_structure),
            labels=price_structure.index,
            colors=palette,
            figure=fig_price_structure)
        
        fig_price_structure.update_layout(
            margin=dict(t=0, b=25, r=0),
            xaxis=dict(
                showspikes=False,
                ticksuffix=' %',
            ),
            yaxis=dict(
                visible=False,
            ),
            barmode='stack',
            hovermode='x unified',
            legend=dict(
                entrywidth=320,
                xanchor='left',
                yanchor='top',
                x=-0.065, y=-0.5,
                orientation='h',
            ),
        )

    return fig_prices_food_growth, fig_price_structure

content = html.Div(id='page-content', style={'font-weight':'200'})

app = Dash(
    name='rus',
    title='Экономика России',
    # external_stylesheets=[dbc.themes.BOOTSTRAP],
    # external_stylesheets=['assets/bWLwgP.css'],
    external_stylesheets=['assets/custom.css'],
    suppress_callback_exceptions=True
)
server = app.server

app.layout = html.Div(
    [dcc.Location(id="url"), content],
    style={'font-family': [font_family, 'system-ui', 'Open Sans']}
)

if __name__ == '__main__':
    app.run(debug=True)