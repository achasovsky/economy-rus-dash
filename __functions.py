# Table of Contents:

# 1. Primary functions
# 2. Data functions : manipulations with data
# 3. Date functions : manipulations with dates and datetime
# 4. Plot functions : manipulations with Matplotlib and Seaborn
# 5. Special functions : functions for current notebook

# ----------------------------------------------------------------------------


# Libraries


import numpy as np
import pandas as pd
import scipy
import colorsys
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import collections
import numbers
import decimal
import time
import datetime as dt
import os
import locale
import pickle
import math
import statsmodels
import statsmodels.stats.api as sms

from scipy import stats


# ----------------------------------------------------------------------------


# 1. Primary__ functions


def path_current_get():
    return os.getcwd()


def path_change(path):
    os.chdir(path)


def arange(
        arg1: float,
        arg2: float or None = None,
        arg3: float or None = None,
        arg4: bool or None = None):
    
    '''
    Realization of simple range (based on np.arange) with protection from 
    float large decimals, e.g. 1.100000000009 except 1.1)
    
    default:
        arg1 - start
        arg2 - stop
        arg3 - step
        arg4 - endpoint (if True: 'stop' value included in range; if False: 'stop' value not included in range)

    variations:
        arange(arg1) -> range(start=0, stop=arg1, step=1, endpoint=False)
        
        arange(arg1, arg2):
            arange(float, float) -> (start=arg1, stop=arg2, step=1, endpoint=False)
            arange(float, bool) -> range(start=0, stop=arg1, step=1, endpoint=arg2)
            
            
        arange(arg1, arg2, arg3):
            arange(float, float, float) -> (start=arg1, stop=arg2, step=arg3, endpoint=False)
            arange(float, float, bool) -> range(start=arg1, stop=arg2, step=1, endpoint=arg3)
            
        arange(arg1, arg2, arg3, arg4):
            arange(float, float, float, bool) -> range(start=arg1, stop=arg2, step=arg3, endpoint=arg4)

    dependencies:
        libraries: numpy, decimal, numbers
    '''

    # list of argument values
    arg_values = locals().values()

    # create list with decimals of arguments values
    round_idxs = []
    for i in arg_values:
        if (isinstance(i, numbers.Number) and not
            isinstance(i, bool)):
            decimals = decimal.Decimal(str(i)).as_tuple().exponent
            round_idxs.append(abs(decimals))
    # find maximum number of decimals - 
    # all values would be round to it later to avoid X.XXXXXXXXXX float
    round_dec = max(round_idxs)
    
    # True/False marker if result should be all integers
    is_int = False

    # if only one argument: arange(arg1)
    if ((arg1 is not None) & (arg2 is None) &
        (arg3 is None) & (arg4 is None)):
        # equivalent (start=0, stop=arg1, step=1, endpoint=False)
        start = 0
        stop = arg1
        # return empty array if start and stop equals
        if start == stop:
            arr = np.empty(0)
            return arr
        step = 1
        endpoint = False
        # rememeber decimal number of stop variable
        round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
        round_dec_for_stop = abs(round_dec_for_stop.exponent)
        
        if isinstance(arg1, int):
            is_int = True

    # if two arguments: arange(arg1, arg2)
    if ((arg1 is not None) & (arg2 is not None) &
        (arg3 is None) & (arg4 is None)):
        
        # if second argument boolean: arange(number1, True)
        if isinstance(arg2, bool):
            # equivalent (start=0, stop=arg1, step=1, endpoint=arg2)
            start = 0
            stop = arg1
            step = 1
            endpoint = arg2
            # rememeber decimal number of stop variable
            round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
            round_dec_for_stop = abs(round_dec_for_stop.exponent)
        # if second argument not boolean: arange(number1, number2)
        else:
            # equivalent (start=arg1, stop=arg2, step=1, endpoint=False)
            start = arg1
            stop = arg2
            # return empty array if start and stop equals
            if start == stop:
                arr = np.empty(0)
                return arr
            step = 1
            endpoint = False
            # rememeber decimal number of stop variable
            round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
            round_dec_for_stop = abs(round_dec_for_stop.exponent)

        if isinstance(arg1, int) & isinstance(arg2, int):
            is_int = True

    # if three arguments: arange(arg1, arg2, arg3)
    if ((arg1 is not None) & (arg2 is not None) &
        (arg3 is not None) & (arg4 is None)):
        # if third argument boolean: arange(number1, number2, True)
        if isinstance(arg3, bool):
            # equivalent (start=arg1, stop=arg2, step=1, endpoint=arg3)
            start = arg1
            stop = arg2
            # return empty array if start and stop equals
            if start == stop:
                arr = np.empty(0)
                return arr
            step = 1
            endpoint = arg3
            # rememeber decimal number of stop variable
            round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
            round_dec_for_stop = abs(round_dec_for_stop.exponent)
        # if third argument not boolean: arange(number1, number2, number3)
        else:
            # equivalent (start=arg1, stop=arg2, step=arg3, endpoint=False)
            start = arg1
            stop = arg2
            # return empty array if start and stop equals
            if start == stop:
                arr = np.empty(0)
                return arr
            step = arg3
            endpoint = False
            # rememeber decimal number of stop variable
            round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
            round_dec_for_stop = abs(round_dec_for_stop.exponent)

        if (isinstance(arg1, int) & isinstance(arg2, int) &
               isinstance(arg3, int)):
            is_int = True

    # if all arguments: arange(arg1, arg2, arg4, True)
    if ((arg1 is not None) & (arg2 is not None) &
        (arg3 is not None) & (arg4 is not None)):
        # equivalent (start=arg1, stop=arg2, step=arg3, endpoint=arg4)
        start = arg1
        stop = arg2
        # return empty array if start and stop equals
        if start == stop:
            arr = np.empty(0)
            return arr
        step = arg3
        endpoint = arg4
        # rememeber decimal number of stop variable
        round_dec_for_stop = decimal.Decimal(str(stop)).as_tuple()
        round_dec_for_stop = abs(round_dec_for_stop.exponent)

        if (isinstance(arg1, int) & isinstance(arg2, int) &
            isinstance(arg3, int)):
            is_int = True

    # arr = step * np.arange(start/step, stop/step)
    arr = np.arange(start, stop, step)
    # round array to avoid X.XXXXXXXXXXXX float
    arr = np.around(arr, decimals=round_dec)
    # if last value of arr plus step equals to stop it concatenates to arr
    last_value = arr[-1]
    # also round this value to avoid X.XXXXXXXXXXXX float (number decimals as in stop variable)
    last_value_plus_step = np.around(last_value+step, round_dec_for_stop)
    if endpoint and last_value_plus_step==stop:
        arr = np.concatenate([arr,[stop]])
    if is_int:
        arr = np.around(arr, decimals=0)
        arr = arr.astype(int)

    return arr


def saturate_color(color, saturation=0.75):

    if isinstance(color, str):
        color_rgb = mpl.colors.to_rgb(color)
    else:
        color_rgb = color
    
    color_hls = colorsys.rgb_to_hls(
        color_rgb[0], color_rgb[1], color_rgb[2])
    color_hls_saturated = (
        color_hls[0], color_hls[1], saturation*color_hls[2])
    color_rgb_saturated = colorsys.hls_to_rgb(
        color_hls_saturated[0], color_hls_saturated[1], color_hls_saturated[2])
    
    return color_rgb_saturated


def saturate_palette(palette, saturation=0.75):
    palette_saturated = [saturate_color(i, saturation=saturation) for i in palette]
    return palette_saturated


def alpha_color(color, alpha=0.75):
    
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)
        
    new_color = tuple (x + (1 - x) * (1 - alpha) for x in color)
    return new_color


def alpha_palette(palette, alpha=0.75):
    palette_alphed = [alpha_color(i, alpha=alpha) for i in palette]
    return palette_alphed


def loadit(name, dir='files', create_empty_dict=False):
    '''
    'create_empty_dict' == True --> function will create empty dictionary,
                                    if there is no such file in directory
    '''
    try:
        result = pd.read_pickle(f'{dir}/{name}.pkl')
        return result
    except FileNotFoundError:
        print(f"File '{name}' not found")
        if create_empty_dict:
            result = {}
            print(f"Empty dictonary '{name}' created")
            return result
        else:
            pass


def saveit(file, name, dir='files'):
    # check if dir exists and create it if not
    if not os.path.exists(dir):
        os.mkdir(dir)
    # save file
    filehandler = open(f'{dir}/{name}.pkl', 'wb') 
    pickle.dump(file, filehandler)
    filehandler.close()
    print(f"File '{name}.pkl' saved in directory '{dir}'")


def savefig(name, dir=None, format='both', dpi=100, transparent=True,  figure=None, **kwargs):
    '''
    Saves figure
    '''
    if dir is None:
        dir = os.getcwd()
    # check if dir exists and create it if not
    if not os.path.exists(dir):
        os.makedirs(dir)
    if format == 'both':
        figure.savefig(
            f'{dir}{name}.png',
            transparent=transparent,
            bbox_inches='tight',
            dpi=dpi,
            format='png',
            **kwargs)
        figure.savefig(
            f'{dir}{name}.svg',
            transparent=transparent,
            bbox_inches='tight',
            dpi=dpi,
            format='svg',
            **kwargs)
        
        print(f"Images '{name}.png' and '{name}.svg' successfully saved into '{dir}' directory")
        
    else:
        figure.savefig(
            f'{dir}{name}.{format}',
            transparent=transparent,
            bbox_inches='tight',
            dpi=dpi,
            format=format,
            **kwargs)
        
        print(f"Image '{name}.{format}' successfully saved into '{dir}' directory")


def order_X_y(data, target):
    '''
    Move Target variable column to the end of DataFrame
    '''
    columns = data.columns.tolist()
    columns.append(columns.pop(columns.index(target)))
    df = data[columns].copy()
    
    return df


def rgb_to_hex(x):
    color_hex = matplotlib.colors.to_hex(x)
    return color_hex


def remove_duplicated_whitespaces(x):

    '''
    Remove duplicated whitespaces in x (String variable)
    '''

    return str.join(' ', str(x).split())


def replace_with_dict(x, replace_dict):

    '''
    In argument 'x' replaces all replace_dict keys by replace_dict values
    '''
    
    for key in replace_dict.keys():
        x = x.replace(key, replace_dict[key])
        
    return x


def df_cutted_rows(data, start, end):
    '''
    Cut n=='start' rows at the beginning of DataFrame and 
    n=='end' rows at the end of DataFrame 
    '''
    if end == 0:
        slice_ = (slice(start, None), slice(None, None))
    else:
        # create slice, that cut rows and stay all columns
        slice_ = (slice(start, -end), slice(None, None))
    # unpack slice_ in .iloc
    df = data.iloc[*slice_].copy()

    return df


def last_row_to_first(data):

    '''
    Make the last row of DataFrame to be the first
    '''

    df = data.copy()
    # extract last row with 'Год' from 'pci_month'
    first_row = df.iloc[-1].to_frame().T
    # add it as first row to 'pci_month'
    df = pd.concat([first_row, df], axis=0)
    # remove last row from 'pci_month'
    df = df.iloc[:-1].copy()

    return df


def np_index(array, value):
    '''
    Returns index of Value which is in Array
    '''
    return np.where(array == value)[0][0]


def is_equal(data1, data2):

    if data1.equals(data2):
        print('Equal')
    else:
        # display rows with differences
        data1[~data1.apply(tuple, 1).isin(data2.apply(tuple, 1))]


def to_round(x, scale=1, error='skip'):
    
    '''
    Round x if possible
    '''
    try:
        return round(x, ndigits=scale)
    except TypeError:
        if error == 'type':
            print(f'TypeError: {x}')
        elif error == 'skip':
            pass
        else:
            print("'error' must be 'type' or 'skip'")
            return
        return x


def to_float(x, errors=False):
    '''
    Convert x to Float if possible
    '''
    try:
        return float(x)
    except ValueError:
        return x
        if errors:
            print(f'ValueError: {x}')

def to_int(x, errors=False):
    '''
    Convert x to Int if possible
    '''
    try:
        return int(x)
    except ValueError:
        return x
        if errors:
            print(f'ValueError: {x}')
    except TypeError:
        return x
        if errors:
            print(f'ValueError: {x}')


def to_string(x):
    '''
    Convert x to String if possible
    '''
    try:
        return str(x)
    except ValueError:
        print(f'ValueError: {x}')
        return x
    except TypeError:
        print(f'ValueError: {x}')
        return x


def not_none(x):
    if x is not None:
        return True
    else:
        return False


def put_column_after(data, column_to_move, column_insert_after):

    '''
    Moves 'column_to_move' from its position to the position after 'column_insert_after'

    Before:
     col1 | column_insert_after | col2 | col3 | col4 | column_to_move | col5
    -------------------------------------------------------------------------
    
    After:
     col1 | column_insert_after | column_to_move | col2 | col3 | col4 | col5
    -------------------------------------------------------------------------
    '''

    df = data.copy()
    
    col = df.pop(column_to_move)
    idx = df.columns.get_loc(column_insert_after) + 1
    df.insert(idx, column_to_move, col)

    return df


def save_session(name, directory='sessions'):
    if directory != 'sessions':
        directory = f'sessions/{directory}/'
    else:
        directory = 'sessions/'
    # check if dir exists and create it if not
    if not os.path.exists(directory):
        os.mkdir(directory)
    # save session
    dill.dump_session(directory+name)


def load_session(name, directory='sessions'):
    if directory != 'sessions':
        directory = f'sessions/{dir}/'
    else:
        directory = 'sessions/'
    # load session
    dill.load_session(directory+name)


def dt_column_to_index(data, column, format=None, freq=None, **kwargs):
    df = data.copy()
    df[column] = pd.to_datetime(df[column], format=format, **kwargs)
    df = df.set_index(column)
    df.index.name = None

    if freq:
        df = df.asfreq(pd.infer_freq(df.index))

    return df


def extract_variable(variable, data):
    try:
        var = data[variable].copy()
        
    except AttributeError:
        var = data[variable]
    except (TypeError, KeyError) as e:
        var = None
        print(f'Variable {variable} not found')

    return var


def stopwatch_start():
    time_start = time.time()
    return time_start


def stopwatch_stop(start, seconds=False):
    if seconds:
        result = time.time() - start
    else:
        result = time.time() - start
        result = dt.timedelta(seconds=np.round(result))
        result = str(result)
    print(f'Execution time: {result}')


def clear_output():
    from IPython.display import clear_output
    clear_output()


# ----------------------------------------------------------------------------


# 2. Data functions


def smoothed(x, n=1000, k=3, y=None, return_type='df', datetime_index=False):
    '''
    Smooth data for plots
    
    Arguments:
    x: pd.DataFrame, pd.Series
    y: array-type
    n: length of linespace
    k: smoothing scale
    return_type: 
        - if 'array' - return x_new, y_new
        - if 'dict' - returns dict with {'x': x_new, 'y': y_new}

    If x == pd.DataFrame functon returns pd.DataFrame anyway

    Libraries:
    from scipy.interpolate import make_interp_spline, BSpline
    '''

    if datetime_index:
        start = x.index[0]
        end = x.index[-1]
        time_range = \
            pd.date_range(start=start, end=end, periods=n)
        x = x.reset_index(drop=True)

    if isinstance(x, pd.DataFrame):
        var_name = x.columns[0] if x.columns[0] != 0 else 'variable'
        x_index = x.index
        x_new = np.linspace(x_index.min(), x_index.max(), n)
        df = pd.DataFrame(index=x_new, columns=x.columns)
        for col in x.columns:
            y = x[col]
            spl = scipy.interpolate.make_interp_spline(x_index, y, k=k)  # type: BSpline
            y_new = spl(x_new)
            df[col] = y_new
        if return_type == 'df':
            if datetime_index:
                df.index = time_range
            return df
        if return_type == 'array':
            return np.array(df.index), np.array(df.iloc[:, 0])
        
    elif isinstance(x, pd.Series):
        var_name = x.name
        y = x.copy()
        x = x.index
        
        # n represents number of points to make between T.min and T.max
        x_new = np.linspace(x.min(), x.max(), n) 
    
        spl = scipy.interpolate.make_interp_spline(x, y, k=k)  # type: BSpline
        y_new = spl(x_new)
    
        if return_type == 'dict':
            if datetime_index:
                ret_dict = {
                    'x': time_range,
                    'y': y_new
                    }
            else:
                ret_dict = {
                    'x': x_new,
                    'y': y_new
                    }
            return ret_dict
        elif return_type == 'array':
            if datetime_index:
                return time_range, y_new
            else:
                return x_new, y_new
        elif return_type == 'df':
            if datetime_index:
                df = pd.DataFrame(data=y_new, index=time_range, columns=[var_name])
            else:
                df = pd.DataFrame(data=y_new, index=x_new, columns=[var_name])
            return df
    else:
        y = x.copy()
        x = arange(len(x))

        # n represents number of points to make between T.min and T.max
        x_new = np.linspace(x.min(), x.max(), n) 
    
        spl = scipy.interpolate.make_interp_spline(x, y, k=k)  # type: BSpline
        y_new = spl(x_new)
        
        if return_type == 'dict':
            if datetime_index:
                ret_dict = {
                    'x': time_range,
                    'y': y_new
                    }
            else:
                ret_dict = {
                    'x': x_new,
                    'y': y_new
                    }
            return ret_dict
        elif return_type == 'array':
            if datetime_index:
                return time_range, y_new
            else:
                return x_new, y_new
        elif return_type == 'df':
            if datetime_index:
                df = pd.DataFrame(data=y_new, index=time_range, columns=['variable'])
            else:
                df = pd.DataFrame(data=y_new, index=x_new, columns=['variable'])
            return df


def is_nan(df):
    ret = df[df.isna().any(axis=1)]
    shape = df[df.isna().any(axis=1)].shape
    if shape[0] > 0:
        return ret
    else:
        print("No NaN values in DataFrame")


def data_describe(data):
    
    df = data.copy()
    # varibles types
    dtypes = df.dtypes.rename('Type').to_frame()
    # frequency
    frequency = df.count().rename('Count').to_frame()
    # unique values
    unique = df.nunique().rename('Unique').to_frame()
    # NaNs
    nans = df.isnull().sum().rename('NaN').to_frame()
    # NaNs fraction
    nans_frac = df.isnull().mean().round(2)
    nans_frac = nans_frac.rename('Percentages').to_frame()
    # list with results
    results_list = [dtypes, frequency, unique, nans, nans_frac]
    # df with results
    results = pd.concat(results_list, axis=1)
    results['Percentages'] = (results['Percentages'] * 100).astype('int64')
    results = results.sort_values(['NaN'], ascending=False)
    
    return results


def ci_bootstrap(
        data, statistic=np.mean, n_bootstrap=1000,
        confidence_level=0.95, random_state=42):
    '''
    Returns: dict(statistic, std, ci_min, ci_max, margin)
    '''
    data_ = (data,)
    bootstrap = scipy.stats.bootstrap(
        data=data_,
        statistic=statistic,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state
    )
    ci_min = bootstrap.confidence_interval[0]
    ci_max = bootstrap.confidence_interval[1]
    if isinstance(data, pd.DataFrame):
        stat = data.apply(statistic)
        stat = np.array(stat)
        std = np.array(np.std(data, ddof=1))
    else:
        stat = statistic(data)
        std = np.std(data, ddof=1)
    margin = stat - ci_min

    return_dct = {
        'statistic': stat,
        'std': std,
        'ci_min': ci_min,
        'ci_max': ci_max,
        'margin': margin,
    }
    return return_dct


def ci_t_distribution(
        data=None, mean=None, std=None, n=None, confidence_level=0.95):

    if data is not None:
        arr = np.array(data)
        n = len(arr)
        mean = np.mean(arr)
        se = scipy.stats.sem(arr)
        
    if mean and std and n is not None:
        se = std / np.sqrt(n)

    t = scipy.stats.t.ppf((1+confidence_level) / 2, n-1)
    margin = t * se
    ci_min = mean - margin
    ci_max = mean + margin

    return_dct = {
        'mean': mean,
        'ci_min': ci_min,
        'ci_max': ci_max,
        'margin': margin,
        't': t
    }
    return return_dct


def test_normality(data, alpha=0.05):
    
    tests_names = []
    pvalue = []
    condition = []
        
    # Kolmogorov-Smirnov
    ks = stats.kstest(data, 'norm')
    pvalue_ks = ks.pvalue
    tests_names.append('Kolmogorov-Smirnov')
    pvalue.append(pvalue_ks)
    if pvalue_ks < alpha:
        condition.append('Not normal')
    else:
        condition.append('Normal')

    # Anderson-Darling
    and_dar = stats.anderson(data, dist='norm')
    and_dar_sign = and_dar.critical_values[2]
    and_dar_statistic = and_dar.statistic
    tests_names.append('Anderson-Darling (s)')
    pvalue.append(and_dar_statistic)
    if and_dar_statistic > and_dar_sign:
        condition.append('Not normal')
    else:
        condition.append('Normal')

    # Shapiro-Wilk
    pvalue_sw = stats.shapiro(data).pvalue
    tests_names.append('Shapiro-Wilk')
    pvalue.append(pvalue_sw)
    if pvalue_sw < alpha:
        condition.append('Not normal')
    else:
        condition.append('Normal')

    # jarque-bera test
    jb_name = ["Jarque-Bera", "Chi^2", "Skew", "Kurtosis"]
    jb_statistic = sms.jarque_bera(data)
    jb = dict(zip(jb_name, jb_statistic))
    pvalue_jb = jb['Chi^2']
    tests_names.append('Jarque-Bera')
    pvalue.append(pvalue_jb)
    if pvalue_jb < alpha:
        condition.append('Not normal')
    else:
        condition.append('Normal')
    
    # D’Agostino and Pearson
    dagp = stats.normaltest(data)
    pvalue_dagp = dagp.pvalue
    tests_names.append('D’Agostino-Pearson')
    pvalue.append(pvalue_dagp)
    if pvalue_dagp < alpha:
        condition.append('Not normal')
    else:
        condition.append('Normal')

    pvalue = [np.round(i, 4) for i in pvalue]
    results_df = pd.DataFrame({
        'Test': tests_names,
        'P or Statistic (s)': pvalue,
        'Condition': condition,
    })
    
    return results_df


def feature_importance_display(
        features, importance,
        top=None, imp_min_level=None, only_features=True):

    '''
     
    '''

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    if imp_min_level is not None:
        loc_row = feature_importance['Importance'] > imp_min_level
        feature_importance = (feature_importance
                              .loc[loc_row, :]
                              .sort_values('Importance', ascending=False)
                              .reset_index(drop=True))
    if top is not None:
        feature_importance = (feature_importance
                             .sort_values('Importance', ascending=False)
                             .reset_index(drop=True))
        feature_importance = feature_importance.loc[0:top-1]

    if only_features:
        feature_importance = feature_importance['Feature']
        
    return feature_importance


def outliers_column_iqr(data, feature, scale=1.5):

    '''
    Add nominative (1/0) column '{feature}_is_out' in DataFrame, that indicates outliers for Feature
    '''

    df = data.copy()

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_boundary = q1 - scale*iqr
    upper_boundary = q3 + scale*iqr
    condition = ((df[feature] < lower_boundary) |
                 (df[feature] > upper_boundary))
    df[feature+'_is_out'] = condition.astype(int)

    return df


def correlation_w_target(data, target):

    '''
    Create sorted DataFrame with correlations to Target 
    '''
    
    df = (data
          .corr()[target]
          .sort_values(ascending=False, key=abs)[1:]
          .to_frame())
    return df


def check_columns_match(data):

    '''
    Check if all columns in DataFrame are equal and return no equal if not
    '''

    df = data.copy()
    df['is_equal'] = df.eq(df.iloc[:, 0], axis=0).all(1).astype(int)
    equal_sum = df['is_equal'].sum()

    if equal_sum == len(df):
        print('All values matched')
        return None
    else:
        loc = df['is_equal'] == 0, df.columns != 'is_equal'
        result = df.loc[loc].copy()
        return result      


def fillna_na(data, features_list):

    '''
    Fill all NaNs in DataFrame by 'NA'
    '''

    df = data.copy()
    for feature in features_list:
        df[feature] = df[feature].fillna('NA')

    return df


def normalized_by_first(data, return_type='df'):

    '''
    Normalize kind: 
        first_value == first_value
        second_value = second_value / first_value
        third_value = third_value / first_value
    '''
    
    first_value = data[0]
    
    data_new = [(x/first_value) for x in data]

    if return_type == 'df':
        df = pd.DataFrame(data=data_new, index=data.index)
        return df
    if return_type == 'series':
        series = pd.Series(data=data_new, index=data.index)
        return series
    elif return_type == 'array':
        array = np.array(data_new)
        return array
    elif return_type == 'list':
        lst = list(data_new)
        return lst
    else:
        print("'return_type' must be 'df', 'series', 'array', 'list'")
    
    return data_new


def normalized(data, reshape=True, return_type='df'):

    '''
    MinMaxScaler 0/1 
    '''
    
    if (isinstance(data, pd.Series) | 
        isinstance(data, pd.DataFrame)):
        idxs = data.index.copy()
    if reshape:
        data = np.array(data).reshape(-1, 1)
    data_new = MinMaxScaler().fit_transform(data)
    if return_type == 'df':
        data_new = pd.DataFrame(data=data_new, index=idxs)
    elif return_type == 'array':
        pass
    else:
        print("return_type must be 'df' or 'array'")
        return None
        
    return data_new


def skewness(df):

    df = pd.DataFrame(df.skew(numeric_only=True),
                      columns=['Skewness'],
                      index=None)

    df['Highly skewed'] = (abs(df['Skewness']) > 0.5)
    df['abs'] = abs(df['Skewness'])

    df = df.sort_values(by=['abs', 'Highly skewed'], ascending=False)
    df = df.drop('abs', axis=1)

    return df


def kurtosis(df):

    df = pd.DataFrame(df.kurtosis(numeric_only=True),
                      columns=['Kurtosis'],
                      index=None)
    df['Type'] = np.nan

    df.loc[df['Kurtosis'] > 1, 'Type'] = 'Too Peaked'
    df.loc[df['Kurtosis'] < -1, 'Type'] = 'Too Flat'
    df.loc[(df['Kurtosis'] <= 1) & (df['Kurtosis'] >= -1), 'Type'] = 'Normal'
    
    df['abs'] = abs(df['Kurtosis'])
    df = df.sort_values(by=['abs', 'Type'], ascending=False)
    df = df.drop('abs', axis=1)

    return df


def plot_acf(
        acf_w_alphas=None, data=None, lags=40, partial=False, scatter=False, s=2,
        transparency_lines=1, color_lines=None, exclude_first=True,
        transparency_significant=0.15, show_last_significant=True,
        last_significant_delta=0.1, color_significant=None, color_annotate=None, **kwargs):

    if acf_w_alphas is None:
        acf_w_alphas = ts_acf_calculate(data, lags=lags, partial=partial, **kwargs) 
        
    acf = acf_w_alphas[:, 0]
    alphas = acf_w_alphas[:, 1:]
    
    lags = len(acf)
    xticks = arange(lags)
    color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color_significant = color_significant or color_palette[2]
    color_lines = color_lines or color_palette[0]

    color_annotate = color_annotate or color_palette[1]

    if exclude_first:
        acf[0] = 0
        alphas[:1] = 0

    if scatter:
        plt.scatter(
            x=xticks,
            y=acf,
            s=s
        )
    for i in arange(lags):
        plt.plot(
            [i, i],
            [0, acf[i]],
            color=color_lines,
            alpha=transparency_lines
        )
    if exclude_first:
        plt.fill_between(
            arange(lags)[1:],
            (alphas[:, 0] - acf)[1:],
            (alphas[:, 1] - acf)[1:],
            lw=0,
            color=color_significant,
            alpha=transparency_significant
        )
    else:
        plt.fill_between(
            arange(lags),
            alphas[:, 0] - acf,
            alphas[:, 1] - acf,
            lw=0,
            color=color_significant,
            alpha=transparency_significant
        )

    if show_last_significant:
        last_sign = ts_acf_last_significant_index(data=data, partial=partial)
        pacf_text = f'{last_sign}'
        last_sign_y = acf[last_sign] + last_significant_delta
    
        plt.annotate(
            text=pacf_text,
            xy=(last_sign, last_sign_y),
            ha='center',
            size=9,
            color=color_annotate,
            weight='bold')

    plt.plot([-1, lags], [0, 0])
    plt.gca().spines[['bottom', 'left']].set_visible(False)
    plt.grid(False)
    plt.xlim(-2, lags+1)


def ts_acf_calculate(data, lags=36, alpha=0.05, partial=False, **kwargs):

    if partial:
        acf_result = statsmodels.tsa.stattools.pacf(
            data, nlags=lags, alpha=alpha, method='ywadjusted', **kwargs)
    else:
        acf_result = statsmodels.tsa.stattools.acf(
            data, nlags=lags, alpha=alpha, missing='none', **kwargs)

    acf = acf_result[0]
    alphas = acf_result[1]
    result = np.hstack([acf.reshape(-1,1), alphas])
    
    return result


def ts_acf_last_significant_index(data, lags=36, partial=False):
    '''
    Return index of first insignificant element in ACF or PACF

    Attributes:
        ci - confident intervals for ACF value (example, result[1] of statsmodels.tsa.stattools.acf)
    '''
    acf = ts_acf_calculate(data, lags=lags, partial=partial)
    ci = acf[:, 1:]
    
    # for i, j in enumerate(ci):
    #     status = np.all(j > 0) if j[0] > 0 else np.all(j < 0)
    #     if not status:
    #         break
    for i, j in enumerate(ci):
        # check if values in 'alphas' have not equal sign
        if ((j[0]<0) != (j[1]<0)):
            return i-1
        elif i == len(ci)-1:
            print(f'All {lags} lags significant')


def ts_arima_forecast(model, steps, data, ci=[80, 95]):

    df = data.copy()
    results = model.get_forecast(steps=steps)

    final_df = pd.DataFrame(
        index = pd.date_range(
            df.index[0], results.predicted_mean.index[-1], freq=df.index.freq),
        data=pd.concat([
            df.iloc[:,0], results.predicted_mean], axis=0),
        columns=['data'])

    
    final_df['forecast'] = np.where(
        final_df.index.date < results.predicted_mean.index[0].date(), 0, 1)

    for ci_value in ci:
        alpha = (100 - ci_value) / 100
        final_df[f'lower_ci{ci_value}'] = \
            results.conf_int(alpha=alpha).iloc[:, 0]
        final_df[f'upper_ci{ci_value}'] = \
            results.conf_int(alpha=alpha).iloc[:, 1]

    return final_df


def test_poisson_bootstrap(
        data1, 
        data2,
        n_bootstrap=10000,
        ci=[2.5,97.5],
        decimals = 3,
        plot=True,
        figsize=(7, 2),
        colors=None,
        execution_time=True,
        results_dict=False,
        means_plots=True,
        rstyle=True,
        rstyle_dataplot_kwargs={},
        rstyle_meansplot_kwargs={},
        simple_results=False):

    '''
    If plot == True and results_dict == True and means_plots == True
    Returns: 
        - dict with means difference value and boundaries
        - dict with Poisson bootstrap folds
        - figure with means diffrenece plot with boundaries
        - figure with data plots
    '''

    if simple_results:
        plot = False
        execution_time = False
        results_dict = False
        means_plots = False
    
    t_start = time.time()

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color0 = colors[0]
    color1 = colors[1]
    color_red = saturate_color('#CD4A3F', 0.9)
    color_grey = '#5B5B5B'
    color_grey_dark = '#505050'

    if not isinstance(data1, np.ndarray):
        data1 = np.array(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    means_diff = mean1 - mean2

    poisson_bootstraps1 = stats.poisson(1).rvs(
        (n_bootstrap, len(data1))).astype(np.int64)

    poisson_bootstraps2 = stats.poisson(1).rvs(
        (n_bootstrap, len(data2))).astype(np.int64)

    mean1_boot = (poisson_bootstraps1*data1).sum(axis=1) / len(data1)
    mean2_boot = (poisson_bootstraps2*data2).sum(axis=1) / len(data2)
    means_diff_boot = mean1_boot - mean2_boot
    
    lower_boundary, upper_boundary = np.percentile(means_diff_boot, ci)
    
    means_dict = {
        'mean1': mean1_boot,
        'mean2': mean2_boot,
        'means_diff': means_diff_boot
    }
    
    results = {
        'Lower Boundary': lower_boundary, 
        'Means Difference': means_diff, 
        'Upper Boundary': upper_boundary
    }

    if not simple_results:
        if not plot:
            print('\n'+'         Poisson bootstrap summary')

    if plot:

        fig_data = plt.figure(figsize=figsize)

        ax = sns.histplot(
            means_diff_boot,
            color=color_grey, alpha=0.4)

        ylim = ax.get_ylim()[1]
        
        ax.vlines(
            0, 0, ylim*0.1,
            color=color_red, linewidth=2.5)
        ax.vlines(
            lower_boundary, 0, ylim*0.15,
            color=color_grey_dark, linewidth=1.5)
        ax.vlines(
            upper_boundary, 0, ylim*0.15,
            color=color_grey_dark, linewidth=1.5) 

        ax.set_ylabel('Count')

        if rstyle:
            axis_rstyle(**rstyle_dataplot_kwargs)

        ax.legend(
            **legend_inline(),
            **legend_create_handles(
                3, ['r', 'l', 'l'],
                colors=[color_grey, color_red, color_grey_dark],
                alphas=[0.4, 1, 1],
                labels=['Means difference', 'Zero', 'Significance borders'],
                linelength=1
            ))

        ax.set_title('Poisson bootstrap summary', size=11, pad=27)
        
        plt.show()
        
    # the boundaries, measured by 1 and 99 percentiles,
    # are equvivalent of p-value probabiblities boundaries an 0.05 significant level;
    # if difference in means is out of boundaries range, we reject null hypotesis - 
    # it means that the difference if statistical significant
    if lower_boundary < 0 < upper_boundary:
        significancy = False
    else: 
        significancy = True
    
    # check with Kolmogorov–Smirnov test if distribution of p-values is normal
    # (previously standardize means differences with stats.zscore)
    pvalue_ks = stats.kstest(stats.zscore(means_diff_boot), stats.norm.cdf).pvalue
    
    # Kolmogorov–Smirnov test null hypotesis: distribution of simulation pvalues is normal
    # if pvalue due Kolmogorov–Smirnov test <= 0.05, 
    # we reject null hypotesis that distribution of pvalues due simulation is normal;  
    if pvalue_ks <= 0.05:
        distribution = 'not '
    else:
        distribution = ''

    if not simple_results:

        mean1_rnd = f"%.{decimals}f" % mean1
        mean2_rnd = f"%.{decimals}f" % mean2
        lower_boundary_rnd = f"%.{decimals}f" % lower_boundary
        means_diff_rnd = f"%.{decimals}f" % means_diff
        upper_boundary_rnd = f"%.{decimals}f" % upper_boundary
        pvalue_rnd = f"%.{decimals}f" % pvalue_ks
        
        ha1 = '===================================================================================='
        start1 = '          '
        space = '              '

        delta1 = 27 - len(f'Lower Boundary:{lower_boundary_rnd}')
        delta1 = delta1 * ' '
        delta2 = 27 - len(f'Means Difference:{means_diff_rnd}')
        delta2 = delta2 * ' '
        delta3 = 27 - len(f'Upper Boundary:{upper_boundary_rnd}')
        delta3 = delta3 * ' '

        delta4 = 43 - len(f'Significantly difference:{significancy}')
        delta4 = delta4 * ' '
        delta5 = 43 - len(f"Means Differences' distribution:{distribution}normal")
        delta5 = delta5 * ' '
        delta6 = 43 - len(f'Kolmogorov–Smirnov test p-value:{pvalue_rnd}')
        delta6 = delta6 * ' '

        print(
            '\n'
            f'{start1}'f'Significantly difference:{delta4}\033[1m{significancy}\033[0m' \
                + space + f'Lower Boundary:{delta1}{lower_boundary_rnd}' '\n' 
            f'{start1}'f"Means Differences' distribution:{delta5}{distribution}normal" \
                + space + f'Means Difference:{delta2}{means_diff_rnd}' '\n' 
            f'{start1}'+f'Kolmogorov–Smirnov test p-value:{delta6}{pvalue_rnd}' \
                + space+ f'Upper Boundary:{delta3}{upper_boundary_rnd}' '\n' '\n' 
            f'{start1}' + ha1 + '\n' '\n' \
            f'{start1}' + f'Sample 1 mean: {mean1_rnd}' '\n'
            f'{start1}' + f'Sample 2 mean: {mean2_rnd}')

    if means_plots:
        
        fig_means = plt.figure(figsize=figsize)
        
        ax = sns.histplot(
            mean1_boot,
            color=color0, alpha=0.5)
        
        ax = sns.histplot(
            mean2_boot, 
            color=color1, alpha=0.5)
        
        ax.set(xlabel=None)
        ax.set_ylabel('Count', weight='bold')
        ax.set_xlabel('Sample Means')
        
        ylim = ax.get_ylim()[1]
        
        ax.vlines(
            np.mean(mean1_boot), 0, ylim,
            color=saturate_color(color0, 1.25), linewidth=0.75, ls='--')
        ax.vlines(
            np.mean(mean2_boot), 0, ylim,
            color=saturate_color(color1, 1.25), linewidth=0.75, ls='--')

        if rstyle:
            axis_rstyle(**rstyle_meansplot_kwargs)

        ax.legend(
            **legend_inline(),
            **legend_create_handles(
                2, 's',
                colors=[color0, color1],
                alphas=[0.5, 0.5],
                labels=['Sample 1', 'Sample 2']))

        plt.show()

    if execution_time:
        
        execution_time = np.round(time.time() - t_start, 2)
        execution_time_formated = \
                         str(dt.timedelta(seconds=np.round(time.time() - t_start)))
        
        print(f'{start1}'+'Execution time: {}'.format(execution_time_formated))
        print(f'{start1}'+'Execution time (seconds): {}'.format(execution_time, '\n'))

    if results_dict:
        return results, means_dict, fig_data, fig_means

    if simple_results:
        return significancy


# ----------------------------------------------------------------------------


# 3. Date functions


def to_date(x, kind='%B %Y', translate=False):
    '''
    String to Date
    '''
    months_list = [
        'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль',
        'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь',
        'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль',
        'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'
    ]
    # if months in Russian
    if translate:
        # split string to list
        x = x.split()
        # for every element in list
        for i in x:
            # if element is month
            if i in months_list:
                # find its index
                i_index = x.index(i)
                # translate element and access new value with it
                new_value = months_translate(i, kind='rus-eng', capitalize=True)
                # change old month to new one
                x[i_index] = new_value
        # join all elements of list to one string
        x = ' '.join(x)
    # transform string to date
    x = dt.datetime.strptime(x, kind)
    return x


def months_translate(x, kind='rus-eng', add_year=None, capitalize=True):

    '''
    Transform russian month name to english
    'январь' --> 'January'
    
    if add_year==2021: 'январь' --> 'January 2021'
    if capitalize==False: 'январь' --> 'january'
    '''
    
    # lowercase data
    x_old = x.lower()
    # create repalce dict
    if kind == 'rus-eng':
        repalce_dict = {
            'январь': 'january',
            'февраль': 'february',
            'март': 'march',
            'апрель': 'april',
            'май': 'may',
            'июнь': 'june',
            'июль': 'july',
            'август': 'august',
            'сентябрь': 'september',
            'октябрь': 'october',
            'ноябрь': 'november',
            'декабрь': 'december'
        }
    elif kind == 'eng-rus':
        repalce_dict = {
            'january': 'январь',
            'february': 'февраль',
            'march': 'март',
            'april': 'апрель',
            'may': 'май',
            'june': 'июнь',
            'july': 'июль',
            'august': 'август',
            'september': 'сентябрь',
            'october': 'октябрь',
            'november': 'ноябрь',
            'december': 'декабрь'
        }
    else:
        print("'kind' must be 'rus-eng' or 'eng-rus'")
    # for all keys and values in dict, replace x by value if x and key are equal
    for k, v in repalce_dict.items():
        if x_old == k:
            x_new = v
        else:
            pass

    if capitalize:
        x_new = x_new.capitalize()

    if add_year is not None:
        x_new = x_new + ' ' + str(add_year)

    return x_new


def set_location(loc='EN'):
    if loc=='EN':
        locale.setlocale(locale.LC_ALL,'en_US')
    elif loc=='RU':
        locale.setlocale(locale.LC_ALL,'ru_RU.UTF-8')
    else:
        print("Location have to be 'EN' or 'RU'")


# ----------------------------------------------------------------------------


# 3. Plot__ functions


def plot_fill_between(x, y1, y2, color, alpha=0.1, ax=None, **kwargs):
    
    if ax is None:
        plt.fill_between(
            x, y1, y2,
            interpolate=True, color=color, ec='none', alpha=alpha, **kwargs)
    else:
        ax.fill_between(
            x, y1, y2,
            interpolate=True, color=color, ec='none', alpha=alpha, **kwargs)


def plot_timemarker(
        text, x, y_text, y_line, delta, y_min=0, color_text=None, color_scatter='#AF4035',
        ha='left', weight='bold', size=8, show=None, ax=None, **kwargs):
    
    if ha == 'right':
        delta = -delta

    if ax is None: ax = plt.gca()
 
    # point
    ax.scatter(
    x=x,
    y=y_text, color=color_scatter, s=5, zorder=6)
    # line
    ax.axvline(
        x=x,
        ymin=y_min, ymax=y_line, lw=0.85, ls=':',
        color=color_scatter, alpha=0.75, zorder=0)
    # text
    x_text = x + delta
    ax.text(
        x=x_text,
        y=y_text, s=text, ha=ha, va='center', weight=weight,
        size=size, color=color_text, alpha=1, **kwargs)

    if show is None:
        pass
    else:
        plt.show()


def axis_rstyle(
        yticks: list | None = None,
        xticks: list | None = None,
        yslice: list | None = None,
        xslice: list | None = None,
        ylim: list | None = None,
        xlim: list | None = None,
        x_spine_lim: list | None = None,
        y_spine_lim: list | None = None,
        x_axis_hide: bool = False,
        y_axis_hide: bool = False,
        x_ticks_hide: bool = False,
        y_ticks_hide: bool = False,
        x_ticklabels_hide: bool = False,
        y_ticklabels_hide: bool = False,
        offset_left: float = 5,
        offset_bottom: float = 5,
        ticks_pad_left: float = 6,
        ticks_pad_bottom: float = 6,
        linewidth: float = 0.75,
        margin: bool = True,
        customize_colors: bool = True,
        spines_color: str ='#CCCCCC',
        ticks_color: str ='#CECECE',
        ticklabels_color: str ='#808080',
        grid: bool = False,
        ax=None):
    
    '''
    xticks: tuple (x_min, x_max, step)
    yticks: tuple (y_min, y_max, step)

    Dependencies: 
        import: collections
        functions: arange
    '''
    
    if ax is None: ax = plt.gca()

    # order of steps (important):
        # 1 - get ticks
        # 2 - set margins if necessary
        # 3 - manipulations with sticks
        # 4 - update ticks
        # 5 - spines modification
        # 6 - set limits
        # 7 - tick params
        # 8 - grid

    # get ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    if margin is not None:
        if isinstance(margin, collections.abc.Iterable):
            ax.margins(*margin)
        else:
            margin = 0.01 if margin is True else margin
            # calculate margin coefficients coeff0 and coeff1 the way
            # margins have to be equal
            # 1st step: find size of figure/ax -> figisize (or ax) 
            # size should be like (ax_width, ax_height)
            # 2d step: suggest margin_x should be equals 0.025, then
                # ax_width * margin_x = ax_height * margin_y
                # margin_y = (margin_x * ax_width) / ax_height
            # so, calculated by this way values of margin_x and margin_y 
            # would make both margins equal and NOT depend on figure(or ax) size
            ax_height, ax_width = ax.bbox.height, ax.bbox.width
            margin_y = margin * ax_width / ax_height
            ax.margins(x=margin, y=margin_y)

    # declare xticks and yticks if necessary
    if xticks is not None:
        # if step not specified
        if len(xticks) == 2:
            # define step equals default step
            xstep = x_ticks[1] - x_ticks[0]
            # make xticks shape (3,)
            xticks = np.append(xticks, xstep)
        x_ticks = arange(xticks[0], xticks[1], xticks[2], True)
    if yticks is not None:
        # if step not specified
        if len(yticks) == 2:
            # define step equals default step
            ystep = y_ticks[1] - y_ticks[0]
            # make yticks shape (3,)
            yticks = np.append(yticks, ystep)
        y_ticks = arange(yticks[0], yticks[1], yticks[2], True)

    # declare xticks and yticks with slices if necessary
    if xslice is not None:
        xslice_ = slice(*xslice)
        x_ticks = x_ticks[xslice_]
    if yslice is not None:
        yslice_ = slice(*yslice)
        y_ticks = y_ticks[yslice_]

    # update ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # set limits if necessary
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        x_ticks = [x for x in x_ticks if x <= xlim[1]]
        x_ticks = [x for x in x_ticks if x >= xlim[0]]
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        y_ticks = [y for y in y_ticks if y <= ylim[1]]
        y_ticks = [y for y in y_ticks if y >= ylim[0]]

    # customize spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_bounds(x_ticks[0], x_ticks[-1])
    ax.spines['bottom'].set_position(('outward', offset_bottom))
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_bounds(y_ticks[0], y_ticks[-1])
    ax.spines['left'].set_position(('outward', offset_left))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if x_spine_lim:
        ax.spines['bottom'].set_bounds(x_spine_lim[0], x_spine_lim[-1])
    if y_spine_lim:
        ax.spines['left'].set_bounds(y_spine_lim[0], y_spine_lim[-1])

    if customize_colors:
        ax.spines['bottom'].set_color(spines_color)
        ax.spines['left'].set_color(spines_color)
        ax.tick_params(which='both', color=ticks_color)
        ax.tick_params( which='both', labelcolor=ticklabels_color)

    if linewidth:
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['left'].set_linewidth(linewidth)
        ax.tick_params(which='both', width=linewidth)
    
    # set tick params and colors
    ax.tick_params(
        which='both', direction='out', bottom=True, size=3, left=True)
    ax.tick_params(
        axis='x', pad=ticks_pad_bottom)
    ax.tick_params(
        axis='y', pad=ticks_pad_left)

    if x_axis_hide:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False)
    if x_ticks_hide:
        ax.tick_params(bottom=False)
    if x_ticklabels_hide:
        ax.tick_params(labelbottom=False)
    if y_axis_hide:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
    if y_ticks_hide:
        ax.tick_params(left=False)
    if y_ticklabels_hide:
        ax.tick_params(labelleft=False)

    # grid customization (exclude grid lines at the edge of spines)
    if grid:
        if not isinstance(grid, bool):
            raise TypeError ("'grid' agrument must be Bool")
            
        ax.grid(False)
        x_ticks_ = ax.get_xticks()
        y_ticks_ = ax.get_yticks()

        for i in x_ticks_:
            if (i == x_ticks_[0]) | (i == x_ticks_[-1]):
                pass
            else:
                ax.plot(
                    [i, i], [y_ticks_[0], y_ticks_[-1]],
                    lw=0.5, ls=':', color='#D9D9D9', zorder=-10)
        for i in y_ticks_:
            if (i == y_ticks_[0]) | (i == y_ticks_[-1]):
                pass
            else:
                ax.plot(
                    [x_ticks_[0], x_ticks_[-1]], [i, i],
                    lw=0.5, ls=':', color='#D9D9D9', zorder=-10)
    else:
        ax.grid(False)


def axis_secondary(
        where='bottom',
        pad=27,
        xticks=None,
        xlabels=None,
        label_color='#808080',
        ax=None):

    if ax is None: ax = plt.gca()

    axis_sec =  ax.secondary_xaxis(where)
    axis_sec.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    axis_sec.tick_params(
        bottom=False, top=False, right=False, left=False,
        pad=pad, colors=label_color)

    if xticks is not None:
        axis_sec.set_xticks(ticks=xticks, labels=xlabels)

    return axis_sec


def legend_inline(
        ncols=None,
        loc='lower left',
        bbox_to_anchor=(0,1),
        frameon=False,
        ax=None):

    if ax is None: ax = plt.gca()
    ncols_fact = len(ax.get_legend_handles_labels()[0])

    ncols = ncols or ncols_fact or 6

    params = {
        'ncols': ncols,
        'loc': loc,
        'bbox_to_anchor': bbox_to_anchor,
        'frameon': frameon}
    
    return params


def legend_mid(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(1,1),
        markersize=1,
        labelspacing=0.5,
        alignment='left'):

    params = {
        'frameon': frameon,
        'loc': loc,
        'bbox_to_anchor': bbox_to_anchor,
        'markerscale': markersize,
        'alignment': alignment,
        'labelspacing': labelspacing}
    
    return params


def axis_adjust_barplot(
        axis='x',
        line_hidden=False,
        labelsize=8,
        labelcolor='#808080',
        weight='normal', 
        pad=0,
        ax=None,
        **kwargs):
    
    if ax is None: ax = plt.gca()
        
    if axis == 'x':
        ax.spines['bottom'].set_bounds(
            ax.patches[0].get_x(),
            ax.patches[-1].get_x() + ax.patches[-1].get_width())
        ax.set_xticklabels(ax.get_xticklabels(), weight=weight)

        if line_hidden:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', bottom=False)
        
    if axis == 'y':
        ax.spines['left'].set_bounds(
            ax.patches[0].get_y(),
            ax.patches[-1].get_y() + ax.patches[-1].get_height())
        ax.set_yticklabels(ax.get_yticks(), weight=weight)

        if line_hidden:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)

    ax.tick_params(
        axis=axis, labelsize=labelsize, labelcolor=labelcolor,
        pad=pad, **kwargs) 


def ax_current():
    return plt.gca()


def legend_create_handles(
        n=None,
        kind='l',
        labels=True,
        colors=None,
        alphas=None,
        markersize=3,
        line_linestyle = '-',
        linelength = 1.35,
        linewidth = 1.5,
        rectlength = 1.25,
        squaresize = None,
        pointsize = None,
        rectsize = 5,
        ax = None):

    if ax is None: ax = plt.gca()

    if n is None:
        n = len(ax_current().get_legend_handles_labels()[0])
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    result = {}
    handles = []

    if isinstance(kind, str):
        kind = [kind]*n

    if not isinstance(alphas, list):
        alphas = [alphas]*n

    for k, c, a, _ in zip(kind, colors, alphas, arange(n)):

        if k == 'l':
            lw = linewidth
            marker = None
            linestyle = line_linestyle
            alpha = a or 1
        elif k == 's':
            lw = 1.5
            marker = 's'
            markersize = squaresize or markersize
            linestyle = 'None'
            alpha = a or 1
        elif k == 'p':
            lw = 1.5
            marker = 'o'
            markersize = pointsize or markersize
            linestyle = 'None'
            alpha = a or 1
        elif k == 'r':
            lw = rectsize
            marker = None
            markersize = None
            linestyle = '-'
            alpha = a or 0.75 
        else:
            raise ValueError("'kind' must be 'l', 'r', 's' or 'p'")

        handle_local = mpl.lines.Line2D(
            [], [], marker=marker, markersize=markersize,
            linestyle=linestyle, lw=lw, color=c, alpha=alpha)

        handles.append(handle_local)

    if ((kind == 'p') or ('p' in kind) or
        (kind == 's') or ('s' in kind)):
        result['handletextpad'] = 0
    
    if (kind == 'l') or ('l' in kind):
        result['handletextpad'] = 0.75
        result['handlelength'] = linelength
    
    if (kind == 'r') or ('r' in kind):
        result['handlelength'] = rectlength
        result['columnspacing'] = 2.25
        result['handletextpad'] = 1

    result['handles'] = handles
    
    if labels:
        if labels is True:
            labels = ax_current().get_legend_handles_labels()[1]
        else:
            pass
        result['labels'] = labels

    return result


def add_twinx(
        offset_right=10,
        yticks=None,
        ylim=None,
        colors=['#CCCCCC', '#808080'],
        grid=False,
        ax=None):

    if ax is None: ax = plt.gca()
    ax2 = ax.twinx()
    ax2.spines[['left', 'bottom', 'top']].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_position((('outward'), offset_right))
    ax2.spines['right'].set_color('#CCCCCC')
    ax2.tick_params(
        axis='y', direction='out', size=3,
        color=colors[0], labelcolor=colors[1])
    ax2.grid(grid)

    if yticks:
        ax2.set_yticks(ticks=arange(yticks[0], yticks[1], yticks[2])) 
    if ylim:
        ax2.set_ylim(ylim[0], ylim[1])

    return ax2


def ticklabels_f_modify(label_index, new_label, axis='x', ax=None):

    if ax is None: ax = plt.gca()

    if axis == 'x':
        labels = [i.get_text() for i in ax.get_xticklabels()]
        labels[label_index] = new_label
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
    if axis == 'y':
        labels = [i.get_text() for i in ax.get_yticklabels()]
        labels[label_index] = new_label
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(labels)


def ticklabels_f_remove(idx=-1, ax=None):
    if ax is None: ax = plt.gca()
    ax.xaxis.get_major_ticks()[idx].label1.set_visible(False)


def plot_hline(
        y=0,
        xmin=0.01,
        xmax=0.99,
        lw=0.75,
        ls='--',
        zorder=-9,
        color='#CCCCCC',
        ax=None,
        **kwargs):
    
    if ax is None: ax = plt.gca()
        
    ax.axhline(
        y, xmin=xmin, xmax=xmax, lw=lw, ls=ls,
        color=color, zorder=zorder, **kwargs)


def axis_formatter_locator(
        formatter=None, locator=None, axis='x',
        months_capitalize=False, months_upper=False, ax=None):
    '''
    Formatter e.g.:
        mpl.dates.DateFormatter('%b')
    Locator e.g.:
        mpl.dates.MonthLocator([1,4,7,10])
    '''
    
    if ax is None: ax = plt.gca()

    if isinstance(formatter, str):
        formatter = mpl.dates.DateFormatter(formatter)

    if axis == 'x':
        if formatter is not None:
            if months_capitalize:
                formatter_cap = lambda x, pos: formatter(x, pos).capitalize()
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter_cap))
            elif months_upper:
                formatter_cap = lambda x, pos: formatter(x, pos).upper()
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter_cap))
            else:
                ax.xaxis.set_major_formatter(formatter)
        else:
            pass
        if locator is not None:
            ax.xaxis.set_major_locator(locator)
        else:
            pass
        

    if axis == 'y':
        if formatter is not None:
            ax.yaxis.set_major_formatter(formatter)
        else:
            pass
        if locator is not None:
            ax.yaxis.set_major_locator(locator)
        else:
            pass


def axis_translate_months(language='eng-rus', capitalize=True, ax=None):

    if language=='rus-eng':
        locale.setlocale(locale.LC_ALL,'en_US')
    elif language=='eng-rus':
        locale.setlocale(locale.LC_ALL,'ru_RU.UTF-8')

    if capitalize:
        if ax is None: ax = plt.gca()
        formatter = lambda x, pos: mpl.dates.DateFormatter('%b')(x, pos).capitalize()
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))


def axis_add_date_xaxis(
        formatter=None,
        locator=None,
        offset=20,
        labelcolor=None,
        capitalize=True,
        # dateformat='%Y',
        ax=None):
    # set ax
    if ax is None: ax = plt.gca()

    formatter = formatter or mpl.dates.DateFormatter('%Y')
    locator = locator or mpl.dates.YearLocator()
    labelcolor = labelcolor or '#808080'

    ax_ = ax.secondary_xaxis('bottom')
    ax_.spines['bottom'].set_visible(False)
    ax_.spines['bottom'].set_position(('outward', offset))
    ax_.tick_params(bottom=False)
    ax_.xaxis.set_major_locator(locator)
    ax_.xaxis.set_major_formatter(formatter)
    ax_.tick_params(labelcolor=labelcolor)

    if capitalize:
        function = lambda x, pos: formatter(x, pos).capitalize()
        ax_.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(function))
        ax_.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(function))


# ----------------------------------------------------------------------------


# 5. Special__ functions


def pl_legend_title(title, figure=None):
    figure = figure or fig
    figure.update_layout(legend_title_text=title)


def pl_figure(figsize=(750, 250)):
    
    fig = go.Figure()
    
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1])

    return fig


def pl_plot(plot, figure=None):
    figure = figure or fig
    figure.add_trace(plot)


def pl_title(title, figure=None, size=None, color=None):

    figure = figure or fig
    
    fig.update_layout(
        title=dict(text=title),
        margin=dict(t=55))
    
    if size:
        fig.update_layout(
            title=dict(font=dict(size=size)))
    if color:
        fig.update_layout(
            title=dict(font=dict(color=color)))


def pl_labels(x=None, y=None, figure=None):

    figure = figure or fig
    
    figure.update_xaxes(
        side='right',
        title=dict(
            text=x,
            font=dict(weight='bold'),
            standoff=10))

    figure.update_yaxes(
        title=dict(
            text=y,
            font=dict(weight='bold'),
            standoff=10))


def pl_rstyle(xticks=None, xlim=None, yticks=None, ylim=None):
    
    if xticks:
        if len(xticks) == 2:
            fig.update_xaxes(range=xticks)
        if len(xticks) == 3:
            vals_x = arange(xticks[0], xticks[1], xticks[2], True)
            x_lim_left = vals_x[0] - (vals_x[-1] - vals_x[0]) * 0.02
            x_lim_right = vals_x[-1]
            fig.update_layout(
                xaxis = dict(
                    autorange=False,
                    tickvals=vals_x,
                    # range=[xticks[0], xticks[1]],
                    range=[x_lim_left, x_lim_right],
                    tick0=xticks[0],
                    dtick=xticks[2]))
        if xlim:
            fig.update_xaxes(range=xlim)

    if yticks:
        if len(yticks) == 2:
            fig.update_yaxes(range=yticks)
        if len(yticks) == 3:
            vals_y = arange(yticks[0], yticks[1], yticks[2], True)
            y_lim_left = vals_y[0] - (vals_y[-1] - vals_y[0]) * 0.02
            y_lim_right = vals_y[-1]
            fig.update_layout(
                yaxis = dict(
                    autorange=False,
                    tickvals=vals_y,
                    # range=[yticks[0], yticks[1]],
                    range=[y_lim_left, y_lim_right],
                    tick0=yticks[0],
                    dtick=yticks[2]))
        if ylim:
            fig.update_yaxes(range=ylim)


def pl_savefig(figure=None, name=None, dir=None, config=None, fmt='html'):

    figure = figure or fig
    
    if fmt == 'chart_studio':
        py.plot(figure, filename=name, auto_open=False, config=config)
        print("Figure saved into Chart-studio")
        
    elif fmt == 'html':
        # check if dir exists and create it if not
        if dir is None:
            dir = os.getcwd()
            folder = dir.split(os.sep)[-1]
        else:
            folder = dir
        filename = dir+'/'+name+'.'+fmt
        plotly.offline.plot(figure, filename=filename, config=config, auto_open=False)
        print(f"File '{name+'.'+fmt}' saved into folder '{folder}'")
    
    else:
        # check if dir exists and create it if not
        if dir is None:
            dir = os.getcwd()
            folder = dir.split(os.sep)[-1]
        else:
            folder = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        filename = dir+'/'+name+'.'+fmt
        figure.write_image(filename, engine='kaleido')
        
        print(f"File '{name+'.'+fmt}' saved into folder '{folder}'")

def pl_grid(grid=False, figure=None):

    figure = figure or fig
    if not grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    elif grid == 'x':
        fig.update_yaxes(showgrid=False)
    elif grid == 'y':
        fig.update_xaxes(showgrid=False)


def transform_prices_9_6(data_raw, month, year, insert_column=False):
    
    df = data_raw.copy()
    if insert_column:
        df.insert(loc=0, column='delete', value=df.iloc[:, 0].copy())
    # remove unnecessary columns and rows
    df = df.loc[[2, 4], :].copy()
    df = df.iloc[:, 1:].copy()
    # remove spaces in products
    df.loc[2] = df.loc[2].map(remove_duplicated_whitespaces)
    # create index cell
    df.iloc[0, 0] = 'idx'
    # change 'Российская Федерация' to month and year
    df.iloc[1, 0] = month + ' ' + year
    # make row with loc 2 as columns
    df.columns = df.loc[2]
    df = df.drop(2)
    df.columns.name = None
    # make 'idx' column as index
    df = df.set_index('idx', drop=True)
    df.index.name = None

    return df


def saveit_excel(data, filename, path, sheet):
    
    if not os.path.exists(path):
        os.mkdir(path)
    # create full path from directory (path argument) and filename
    path_ = path + '/' + filename + '.xlsx'
    # if such file exist - append  to it new sheet
    if os.path.exists(path_):
        with pd.ExcelWriter(
            path_,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            data.to_excel(
                excel_writer=writer,
                sheet_name=sheet
            )
            print(f"'{sheet}' sheet created in file '{filename + '.xlsx'}'")
    # if not exist - create new .xlsx
    else:
        data.to_excel(
            excel_writer=path_,
            sheet_name=sheet
        )
        print(f"File '{filename+'.xlsx'}' created")


def saveit_csv(data, filename, path):
    
    if not os.path.exists(path):
        os.mkdir(path)
    # create full path from directory (path argument) and filename
    path_ = path + '/' + filename + '.csv'
    # if such file exist - append  to it new sheet
    data.to_csv(
        path_or_buf = path_
    )
    print(f"File '{filename+'.csv'}' created")


def transform_make_header_from_rows(data, rows_index, names=None):
    
    df = data.copy()
    
    if isinstance(rows_index, list):
        # create list of tuples for multiindex columns
        multi_index_columns = []
    
        for i in df.columns:
            multi_index_tuple = ()
            for index in rows_index:
                multi_index_tuple = multi_index_tuple + (str(df.loc[index, i]),)
            multi_index_columns.append(multi_index_tuple)
    
        df.columns=pd.MultiIndex.from_tuples(multi_index_columns, names=names)
        
    else:
        df.columns = df.loc[rows_index]
    
    df = df.drop(rows_index, axis=0)
    
    return df


def transform_concat_rows_strings(df, column_name='index'):
    '''
    In particular column function concat rows if there are two consecutive strings in rows
    If at least NaN - skip
    | string1 |
    -----------  -> do nothing
    |   NaN   |
    -----------
    | string1 |
    -----------  -> add string2 to string1 and remove row with sring2
    | string2 |
    -----------
    '''
    drop_indexes = []
    drop_indexes1 = [0]
    for i in df[column_name].index:
        if i > 1:
            if (isinstance(df[column_name].loc[i-1], str) &
                isinstance(df[column_name].loc[i-2], str)):
                new_value = df[column_name].loc[i-2] + df[column_name].loc[i-1]
                df.loc[i-2, column_name] = new_value
                drop_indexes.append(i-1)
                drop_indexes1.append(i-2)
    df = df.drop(drop_indexes, axis=0)

    return df, drop_indexes1


def transform_fill_values_by_previous(data, kind='row', row_index=None, column_name=None):
    
    df = data.copy()
    j = np.NaN
    if kind == 'row':
        # go through row by column
        for i in df.columns:
            # if value in row not NaN
            if not pd.isna(df.loc[row_index, i]):
                # remember this value in var j
                j = df.loc[row_index, i]
            else:
                # if value in row equals '-', replace it by remembered value in j
                df.loc[row_index, i] = j
    elif kind == 'column':
        # go through column by index
        for i in df.index:
            # if value in column not NaN
            if not pd.isna(df.loc[i, column_name]):
                # remember this value in var j
                j = df.loc[i, column_name]
            else:
                # if value in column equals '-', replace it by remembered value in j
                df.loc[i, column_name] = j
    else:
        print("Argument 'kind' must be 'row' or 'column'")
        
    return df


def transform_resources(
        data, year, FD_partial_names_list, federal_districts_names_list,
        drop_rows_end=None):
    
    df_raw = data.copy()
    # create slice to remove rows at the end of df
    if drop_rows_end is None:
        slice_ = slice(7, None)
    else:
        slice_ = slice(7, -drop_rows_end)
    # remove rows at the end of the df
    df = df_raw.iloc[:, :4][slice_].copy()
    df['Unnamed: 0'] = [i.strip() for i in df['Unnamed: 0']]
    df['Unnamed: 0'] = [i.strip() for i in df['Unnamed: 0']]
    # replace symbols
    replace_dict_part = {
        '/n': '',
        '\n': ' ',
        'Kемеровская область': 'Кемеровская область',
        'г. Москва': 'Москва',
        'г. Санкт-Петербург': 'Санкт-Петербург',
        'г. Севастополь': 'Севастополь'
    }
    df = df.replace(replace_dict_part, regex=True)
    replace_dict_full = {
        ' -': np.NaN,
        '-': np.NaN,
        '…': np.NaN
    }
    df = df.replace(replace_dict_full)
    # reset index
    df = df.reset_index(drop=True)
    # concatenate federal districts that names are separated in two rows
    df = federal_district_concat(df, 'Unnamed: 0', FD_partial_names_list)
    df = df[~df['Unnamed: 0'].isin(federal_districts_names_list)]
    
    df = df.rename(columns={
        'Unnamed: 0': year,
        'Unnamed: 1': 'Всего',
        'Unnamed: 2': 'Городская местность',
        'Unnamed: 3': 'Сельская местность'
    })
    # some clean
    df = df.set_index(year, drop=True)
    df.index.name = None
    # drop unuseful regions
    drop_list = [
    'в том числе:                     Ханты-Мансийский  автономный округ - Югра',
    'в том числе:                   Ненецкий автономный округ',
    'Ямало-Ненецкий  автономный округ',
    'Тюменская область',
    'Архангельская область',
    ]
    df = df.drop(drop_list, axis=0)
    # transform columns to values in two columns: 'variables' and 'values'
    # rename 'variables' to 'index' becausse it will be index level1
    # rename 'values' to year
    df = df.melt(
        var_name='index',
        value_name=year,
        ignore_index=False)
    # create multiindex
    df = df.set_index([df.index, 'index'], drop=True)
    # remove multiindex names
    df.index.names = (None, None)
    # change order if index level0 as in 'regions_names_list'
    # df = df.reindex(regions_names_list, level=0, axis=0)

    return df


def federal_district_concat(data, column_name, federal_district_list):
    '''
    Concat two rows with federal district names

    Raw:
    | column_name       |
    -----------------------------
    | Южный             | NaN   |
    -----------------------------  -> concat this rows and drop one with NaN
    | федеральный округ | 12345 |
    -----------------------------

    Result:
    ---------------------------------
    | Южный федеральный округ | 12345
    ---------------------------------

    Arguments:
    df,
    column_name - column with regions names
    federal_district_list - list with first name of FD ('Южный', 'Центральный', 'Северо-Западный')
    
    '''
    df = data.copy()
    
    for index in df.index:
        if df.loc[index, column_name] in federal_district_list:
            new_value = (df.loc[index, column_name]
                         + ' '
                         + df.loc[index+1, column_name])
            df.loc[index+1, column_name] = new_value
            df = df.drop(index, axis=0)
            
    return df


def get_data_two_level(data, level0=None, level1=None, indexes=None, kind='column'):
    
    df = data.copy()
    # check 'kind' argument
    if kind == 'index':
        df = df.T
    elif kind == 'column':
        pass
    else:
        print("'kind' argument must be 'column' or 'index'")
    # turn 'level' arguments to slice
    if (level0 is None) & (level1 is None):
        return df
    if level0 is None:
        level0 = slice(level0)
    if level1 is None:
        level1 = slice(level1)
    # adress to levels
    df = df.loc[:, (level0, level1)]
    # drop multiindex level0 if both 'levels' are single
    if isinstance(level0, str) & isinstance(level1, str):
        if isinstance(df, pd.Series):
            df = df.to_frame()
            df.columns = df.columns.droplevel(1)
            df.columns.name = data.columns.names[0]
        else:
            df.columns = df.columns.droplevel(0)
    else:
        if isinstance(level0, str):
            df.columns = df.columns.droplevel(0)
        if isinstance(level1, str):
            df.columns = df.columns.droplevel(1)
    # return 
    if kind == 'index':
        return df.T
    if kind == 'column':
        return df
        