"""

"""
import xgboost as xgb
from xgboost.callback import TrainingCallback
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import wandb
import matplotlib.pyplot as plt
from config import CFG
from xgb_functions import series_to_supervised


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 25)
pd.set_option('display.precision', 2)
pd.options.display.max_colwidth = 25


class WandbCallback(TrainingCallback):
    def __init__(self, period=1):
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            # 'mae' is being logged. Adjust for different metrics
            mae = evals_log['train']['mae'][-1]
            wandb.log({'train-mae': mae})
        return False


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


def diy_cv(model, param_grid, splits, trainX, trainy):
    """
    define logging for cross-validation
    :param model:
    :param param_grid:
    :param splits:
    :param trainX:
    :param trainy:
    :return:
    """
    best_score = float("inf")
    best_params = None
    for params in ParameterGrid(param_grid):
        scores = []
        for train_idx, val_idx in splits.split(trainX):
            clone_model = clone(model)
            clone_model.set_params(**params)
            X_train_fold, y_train_fold = trainX.iloc[train_idx], trainy.iloc[train_idx]
            X_val_fold, y_val_fold = trainX.iloc[val_idx], trainy.iloc[val_idx]
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
    """
    callback function to log evaluation results to W&B
    :param period:
    :param show_stdv:
    :return:
    """
    def callback(env):
        if env.iteration % period == 0:
            wandb.log(
                {
                    "Training MAE": env.evaluation_result_list[0][1],
                    "Validation MAE": env.evaluation_result_list[1][1]
                }
            )
    return callback


def wandb_callback():
    """
    callback function to log evaluation results to W&B
    :return:
    """
    def callback(env):
        for i, eval_result in enumerate(env.evaluation_result_list):
            wandb.log({f"{eval_result[0]}-{eval_result[1]}": eval_result[2]})
    return callback


if __name__ == "__main__":
    CFG = CFG()


    # sweep_id = wandb.sweep(CFG.sweep_config, project=CFG.wandb_project_name)

    config_dict = {
        "n_in": 6,
        "n_test": 30,
        "wandb_project_name": 'electricity_demand_forecasting',
        "wandb_run_name": 'xgboost',
        "param_grid": {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 500, 1000],
        }
    }

    # initialize W&B if CFG.logging=True
    if CFG.logging:
        run = wandb.init(
            project=CFG.wandb_project_name,
            name=f'{CFG.wandb_run_name}_v{CFG.version}',
            config=config_dict,
            job_type='train_model'
        )

    # load data
    df = pd.read_parquet('../data/NSW/nsw_df.parquet')
    df.drop(columns=['daily_avg_actual', 'daily_avg_forecast'], inplace=True)
    print("DataFrame shape:", df.shape)
    print("DataFrame head:\n", df.head())

    data = series_to_supervised(df, n_in=CFG.n_in)  # prepare data
    print("Transformed data shape:", data.shape)

    n_obs = CFG.n_in * len(df.columns)

    # split into input and outputs, with the last CFG.n_test rows for testing
    train, test = train_test_split(data.values, CFG.n_test)
    print(f'train shape: {train.shape}')
    print(f'test shape: {test.shape}')

    trainX, trainy = train[:, :-1], train[:, -1]
    print(f'trainX shape: {trainX.shape}')
    print(f'trainy shape: {trainy.shape}')

    testX, testy = test[:, :-1], test[:, -1]
    print(f'testX shape: {testX.shape}')
    print(f'testy shape: {testy.shape}')

    # further split training into train & val sets

    n_val = int(len(trainX) * 0.1)  # use last 10% of data as validation set
    trainX, valX = trainX[:-n_val], trainX[-n_val:]
    trainy, valy = trainy[:-n_val], trainy[-n_val:]

    # convert the datasets into xgb.DMatrix() format
    dtrain = xgb.DMatrix(trainX, label=trainy)  # train set
    dtest = xgb.DMatrix(testX)  # test set
    dval = xgb.DMatrix(valX, label=valy)  # val set

    # train the model
    if CFG.logging:  # perform W&B logging if CFG.logging=True
        # train the model with W&B callback
        bst = xgb.train(
            CFG.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            callbacks=[WandbCallback()] if CFG.logging else []
        )
    else:
        # train the model without W&B callback
        bst = xgb.train(
            CFG.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50
        )

    print(testy)
    # evaluate model
    yhat = bst.predict(dtest)
    print(yhat)
    error = mean_absolute_error(testy, yhat)

    # assuming yhat is the prediction array and testy is the actual target
    # values from the test set
    actual = testy
    predicted = yhat

    # generate a time index for plotting.
    # since we have 30-minute intervals, this can be represented similarly
    # assuming the test set starts immediately after training and val
    # sets, we can calculate the start date as follows this requires the
    # original df to have a datetime index
    test_start_date = df.index[-len(testy)]  # get the start date for test set

    # generate a date range for the test set
    test_dates = pd.date_range(
        start=test_start_date,
        periods=len(testy),
        freq='30min'
    )

    # plotting
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, actual, label='Actual', marker='.', linestyle='-',
             linewidth=1.0)
    plt.plot(test_dates, predicted, label='Predicted', marker='.',
             linestyle='--', linewidth=1.0)
    plt.title('Test Set: Actual vs Predicted Demand')
    plt.xlabel('Date Time')
    plt.ylabel('Total Demand')
    plt.legend()
    plt.tight_layout()
    # can focus on a smaller time frame for a more detailed view
    # plt.xlim([pd.Timestamp('2021-XX-XX'), pd.Timestamp('2021-XX-XX')])
    plt.show()

    # log test MAE to W&B
    if CFG.logging:
        run.log({"Test MAE": error})
        run.finish()

    plt.close()  # close the plot to free up memory
