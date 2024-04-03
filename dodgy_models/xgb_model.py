"""

"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, ParameterGrid
from sklearn.base import clone
import wandb


class CFG:
    n_in = 6
    n_test = 30
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'xgboost'
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 500, 1000],
    }


wandb.init(
    project=CFG.wandb_project_name,
    name=CFG.wandb_run_name,
    config=vars(CFG)
)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    transform a time series dataset with multiple features into a supervised learning dataset.
    :param data: DataFrame containing the features.
    :param n_in: Number of lag observations as input (X).
    :param n_out: Number of observations as output (y).
    :param dropnan: Boolean whether or not to drop rows with NaN values.
    :return: DataFrame of series transformed into supervised format.
    """
    n_vars = data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (data.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % (data.columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def train_test_split(data, n_test):
    """
    split a univariate dataset into train/test sets
    :param data:
    :param n_test:
    :return:
    """
    return data[:-n_test, :], data[-n_test:, :]


def xgboost_forecast(train, testX):
    """
    fit an xgboost model and make a one-step prediction
    :param train:
    :param testX:
    :return:
    """
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# Define custom logging for cross-validation
def custom_cross_validation(model, param_grid, splits, trainX, trainy):
    best_score = float("inf")
    best_params = None
    for params in ParameterGrid(param_grid):
        scores = []
        for train_idx, val_idx in splits.split(trainX):
            clone_model = clone(model)
            clone_model.set_params(**params)
            X_train_fold, y_train_fold = trainX.iloc[train_idx], trainy.iloc[train_idx]  # Corrected to use .iloc for trainy
            X_val_fold, y_val_fold = trainX.iloc[val_idx], trainy.iloc[val_idx]  # Corrected to use .iloc for trainy
            clone_model.fit(X_train_fold, y_train_fold)
            predictions = clone_model.predict(X_val_fold)
            score = mean_absolute_error(y_val_fold, predictions)
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
        print(f"Params: {params}, Avg MAE: {avg_score}")
    return best_params, best_score


def log_evaluation(period=1, show_stdv=True):
    def callback(env):
        if env.iteration % period == 0:
            wandb.log({"Training MAE": env.evaluation_result_list[0][1], "Validation MAE": env.evaluation_result_list[1][1]})
    return callback


if __name__ == "__main__":
    df = pd.read_csv('./../data/NSW/final_df.csv', index_col=0)
    data = series_to_supervised(df, n_in=CFG.n_in)
    # Correct n_obs definition
    n_obs = CFG.n_in * df.shape[1]
    trainX, trainy = data.iloc[:, :n_obs], data.iloc[:, -1].astype('float')  # Assume the last column after transformation is the target

    # Define the model here
    model = xgb.XGBRegressor(objective='reg:squarederror')

    tscv = TimeSeriesSplit(n_splits=5)

    best_params, best_score = custom_cross_validation(model, CFG.param_grid, tscv, trainX, trainy)

    # Log best parameters and MAE score
    wandb.log({"Best MAE": -best_score, "Best Params": best_params})

    # Instantiate model with best parameters and fit
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(trainX, trainy)

    print("Best parameters found: ", best_params)

    testX, testy = trainX[-CFG.n_test:], trainy[-CFG.n_test:]
    yhat = best_model.predict(testX)
    error = mean_absolute_error(testy, yhat)
    print(f'Test MAE: {error:.3f}')
    wandb.log({"Test MAE": error})

    row = data.iloc[-1, :n_obs].values.reshape(1, -1)  # Use n_obs to correctly slice last row for prediction
    yhat = best_model.predict(row)
    print('Predicted: %.3f' % yhat[0])
    wandb.log({'Predicted Value': yhat[0]})

    wandb.finish()

