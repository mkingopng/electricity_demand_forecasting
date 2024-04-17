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
import os
import shap


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 25)
pd.set_option('display.precision', 2)
pd.options.display.max_colwidth = 25

class CFG:
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'xgboost'
    data_path = './../data/NSW'
    images_path = './../images/xgb'
    models_path = './../trained_models/'
    train = False
    logging = True  # set to True to enable W&B logging
    img_dim1 = 20
    img_dim2 = 10
    n_in = 9  # 6 lag features
    n_test = 336  # 7 days of 30-minute sample intervals
    version = 24  # increment for each new experiment
    sweep_count = 10  # number of sweep runs
    params = {
        'objective': 'reg:squarederror',
        'gamma': 4.592513457496951,  # def 0
        'learning_rate': 0.07984076257805875,  # def 0.1
        'max_depth': 8,
        'min_child_weight': 20,  # def 0.1
        'nthread': 4,  # ?
        'random_state': 42,
        'subsample': 0.8836012456010794,
        'reg_alpha': 0.7863437272577511,
        'reg_lambda': 3.475149811652308,  # def 1
        'eval_metric': ['mae'],
        'tree_method': 'hist'
    }
    sweep_config = {
        "method": "random",
        "parameters": {
            "learning_rate": {
                "min": 0.001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.001,
                "max": 1.0
            },
            "min_child_weight": {
                "min": 1,
                "max": 150
            },
            "early_stopping_rounds": {
                "values": [10, 20, 30, 40]
            },
        }
    }


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
    # load data
    df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))
    df.drop(columns=['daily_avg_actual', 'daily_avg_forecast'], inplace=True)
    # print("DataFrame shape:", df.shape)
    # print("DataFrame head:\n", df.head())

    val_end_date = df.index.max()  # most recent date
    val_start_date = val_end_date - pd.Timedelta(days=7)
    test_end_date = val_start_date - pd.Timedelta(minutes=30)
    test_start_date = test_end_date - pd.Timedelta(days=7)

    df_val = df[df.index > val_start_date]
    df_test = df[(df.index > test_start_date) & (df.index <= test_end_date)]
    df_train = df[df.index <= test_start_date]

    train_supervised = series_to_supervised(df_train, n_in=CFG.n_in)
    test_supervised = series_to_supervised(df_test, n_in=CFG.n_in)
    val_supervised = series_to_supervised(df_val, n_in=CFG.n_in)
    # print("Transformed data shape:", data.shape)

    n_obs = CFG.n_in * len(df.columns)

    # split into input and outputs, with the last CFG.n_test rows for testing
    trainX, trainy = train_supervised.iloc[:, :-1], train_supervised.iloc[:, -1]
    testX, testy = test_supervised.iloc[:, :-1], test_supervised.iloc[:, -1]
    valX, valy = val_supervised.iloc[:, :-1], val_supervised.iloc[:, -1]
    # print(f'train shape: {train.shape}')
    # print(f'test shape: {test.shape}')

    # convert the datasets into xgb.DMatrix() format
    dtrain = xgb.DMatrix(trainX, label=trainy)
    dtest = xgb.DMatrix(testX, label=testy)
    dval = xgb.DMatrix(valX, label=valy)

##############################################################################
    # train the model
##############################################################################

    if CFG.train:
        CFG = CFG()
        # sweep_id = wandb.sweep(CFG.sweep_config, project=CFG.wandb_project_name)

        config_dict = {
            "n_in": 10,
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

        # train the model
        bst = xgb.train(
            CFG.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            callbacks=[WandbCallback()] if CFG.logging else []
        )
        # print(testy)

        # evaluate model
        yhat = bst.predict(dtest)
        # print(yhat)

        error = mean_absolute_error(testy, yhat)
        # print(f'Mean Absolute Error: {error}')

        actual = testy
        predicted = yhat

        bst.save_model(os.path.join(CFG.models_path, 'xgb_model.json'))

        # log test MAE to W&B
        if CFG.logging:
            run.log({"Test MAE": error})
            run.finish()

##############################################################################
    # load saved model and run inference and analysis
##############################################################################
    else:
        bst = xgb.Booster()
        bst.load_model(os.path.join(CFG.models_path, 'xgb_model.json'))

        # evaluate model on validation set
        val_predictions = bst.predict(dval)
        val_error = mean_absolute_error(valy, val_predictions)
        print(f'Validation MAE: {val_error}')

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(valX)

        # create an Explanation object for the first observation in val set
        expl = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=valX.iloc[0],
            feature_names=valX.columns
        )

        # calculate SHAP values for multiple instances in the validation set
        shap_values_multi = explainer.shap_values(valX)

        # Exclude the first variable's SHAP values and corresponding feature
        # Exclude 'FORECAST_DEMAND(t-1)' from SHAP values and valX
        feature_to_exclude = 'FORECASTDEMAND(t-1)'  # Adjust if the feature name is slightly different
        feature_index = valX.columns.get_loc(
            feature_to_exclude)  # Get the index of the feature to exclude

        shap_values_excluded = np.delete(shap_values, feature_index, axis=1)  # Remove SHAP values of the feature
        valX_excluded = valX.drop(columns=[feature_to_exclude])  # Drop the column from the DataFrame

        # Prepare data for the first instance
        shap_values_instance_excluded = shap_values_excluded[0, :]  # SHAP values for the first instance, excluded feature
        valX_instance_excluded = valX_excluded.iloc[0, :]  # Feature data for the first instance, excluded feature

        # Create an Explanation object for the waterfall plot
        expl_excluded = shap.Explanation(
            values=shap_values_instance_excluded,
            base_values=explainer.expected_value,
            data=valX_instance_excluded,
            feature_names=valX_excluded.columns.tolist()
        )

        # prep an Explanation object with excluded data for beeswarm plot
        expl_multi_excluded = shap.Explanation(
            values=shap_values_excluded,
            base_values=explainer.expected_value,
            data=valX_excluded,
            feature_names=valX_excluded.columns.tolist()
        )

        # Ensure that the Explanation object includes all these instances
        expl_multi = shap.Explanation(
            values=shap_values_multi,
            base_values=explainer.expected_value,
            data=valX,
            feature_names=valX.columns
        )

        # summary plot: an overview of feature importance
        shap.summary_plot(shap_values, valX, plot_type="bar")
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_summary_plot.png'))
        plt.close()

        shap.summary_plot(shap_values_excluded, valX_excluded, plot_type="bar")
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_summary_plot_excluded.png'))
        plt.close()

        # visualise how the first feature affects the output
        shap.dependence_plot(0, shap_values, valX)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_dependence_plot_0.png'))
        plt.close()

        shap.dependence_plot(1, shap_values, valX)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_dependence_plot_1.png'))
        plt.close()

        shap.decision_plot(
            explainer.expected_value,
            shap_values[0, :],
            valX.iloc[0, :]
        )
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_decision_plot.png'))
        plt.close()

        # decision plot excluding the first feature
        shap.decision_plot(
            explainer.expected_value,
            shap_values_instance_excluded,
            valX_instance_excluded
        )
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_decision_plot_no_FORECASTDEMAND.png'))
        plt.close()

        shap.waterfall_plot(expl)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_waterfall_plot.png'))
        plt.close()

        # Generate the waterfall plot for the first instance with the excluded feature
        shap.waterfall_plot(expl_excluded)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_waterfall_plot_no_FORECASTDEMAND.png'))
        plt.close()

        shap.plots.beeswarm(expl_multi)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_beeswarm_plot.png'))
        plt.close()

        # Generate the beeswarm plot for the excluded data
        shap.plots.beeswarm(expl_multi_excluded)
        fig = plt.gcf()
        fig.set_size_inches(CFG.img_dim1, CFG.img_dim2)
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_shap_beeswarm_plot_no_FORECASTDEMAND.png'))
        plt.close()

        # log validation MAE to W&B
        # if CFG.logging:
        #     wandb.log({"Validation MAE": val_error})

        # plot validation predictions vs actual
        val_dates = pd.date_range(
            start=df_val.index[0],
            periods=len(valy),
            freq='30min'
        )

        plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
        plt.plot(
            val_dates,
            valy,
            label='Actual',
            marker='.',
            linestyle='-'
        )

        plt.plot(
            val_dates,
            val_predictions,
            label='Predicted',
            marker='.',
            linestyle='--'
        )

        plt.title('Validation Set: Actual vs Predicted Demand')
        plt.xlabel('Date Time')
        plt.ylabel('Total Demand')
        plt.legend()
        plt.tight_layout()
        # plt.savefig(os.path.join(CFG.images_path, 'xgb_val_predictions.png'))
        plt.show()
        plt.close()  # close the plot to free up memory
'https://teams.microsoft.com/v2/'