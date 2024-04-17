"""

"""
import pandas as pd
from .config import CFG
import os
import xgboost as xgb


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


def train_test_split(data, n_test):
    """
    Splits a dataset into training and testing sets for time series data.

    This function takes a dataset array and splits it into two sets: a training
    set containing all but the last `n_test` observations, and a test set
    containing the last `n_test` observations. This split ensures that the
    temporal order of observations is preserved, which is crucial for time
    series forecasting.

    :param data (numpy.ndarray): A 2D array of data where rows correspond to
    observations and columns correspond to variables.
    :param n_test (int): The number of observations from the end of the dataset
    to include in the test set.
    :return tuple: A tuple where the first element is the training set as a
    numpy array and the second element is the test set as a numpy array.

    Example:
    >>> dataset = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> train, test = train_test_split(dataset, 1)
    >>> train
    array([[1, 2, 3], [4, 5, 6]])
    >>> test
    array([[7, 8, 9]])
    """
    return data[:-n_test, :], data[-n_test:, :]


def load_and_preprocess_data():
    """
    Load data, preprocess it, and prepare training, testing, and validation datasets.
    """
    # Load data
    df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))
    df.drop(columns=['daily_avg_actual', 'daily_avg_forecast'], inplace=True)

    # Define the date range for validation and test sets
    val_end_date = df.index.max()  # most recent date
    val_start_date = val_end_date - pd.Timedelta(days=7)
    test_end_date = val_start_date - pd.Timedelta(minutes=30)
    test_start_date = test_end_date - pd.Timedelta(days=7)

    # Split data
    df_val = df[df.index > val_start_date]
    df_test = df[(df.index > test_start_date) & (df.index <= test_end_date)]
    df_train = df[df.index <= test_start_date]

    # Convert to supervised format
    train_supervised = series_to_supervised(df_train, n_in=CFG.n_in)
    test_supervised = series_to_supervised(df_test, n_in=CFG.n_in)
    val_supervised = series_to_supervised(df_val, n_in=CFG.n_in)

    # Split into inputs and outputs
    trainX, trainy = train_supervised.iloc[:, :-1], train_supervised.iloc[:, -1]
    testX, testy = test_supervised.iloc[:, :-1], test_supervised.iloc[:, -1]
    valX, valy = val_supervised.iloc[:, :-1], val_supervised.iloc[:, -1]

    # Convert datasets into xgb.DMatrix format
    dtrain = xgb.DMatrix(trainX, label=trainy)
    dtest = xgb.DMatrix(testX, label=testy)
    dval = xgb.DMatrix(valX, label=valy)

    return dtrain, dtest, dval, trainy, testy, valy, valX