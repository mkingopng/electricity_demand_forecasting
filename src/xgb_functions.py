"""

"""
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, target_var='TOTALDEMAND'):
    """
    frame a time series dataset as a supervised learning dataset, required as
    in input for xgb
    :param data:
    :param n_in:
    :param n_out:
    :param dropnan:
    :param target_var:
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):  # input sequence (t-n, ... t-1)
        cols.append(df.drop(columns=target_var).shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars) if
                  df.columns[j] != target_var]

    # forecast sequence (t+1, ... t+n_out-1), only for predictors, not target
    for i in range(1, n_out):
        cols.append(df.drop(columns=target_var).shift(-i))
        names += [('%s(t+%d)' % (df.columns[j], i)) for j in range(n_vars) if
                  df.columns[j] != target_var]

    # add the target variable column at t (current timestep)
    cols.append(df[[target_var]])
    names.append('%s(t)' % target_var)

    # combine everything
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop records with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg
