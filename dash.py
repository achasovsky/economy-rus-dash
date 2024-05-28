from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import os
import colorsys
import arrow
import locale
import requests
import collections
import numbers
import decimal
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline
from __functions import *

# Load data
data = pd.read_csv('excel-csv/prices-2021-2024.csv', index_col=0)

