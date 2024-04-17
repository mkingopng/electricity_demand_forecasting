"""

"""
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import wandb
from .config import CFG
import os
import numpy as np


class WandbCallback(TrainingCallback):
    def __init__(self, period=1):
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            # 'mae' is being logged. Adjust for different metrics
            mae = evals_log['train']['mae'][-1]
            wandb.log({'train-mae': mae})
        return False


class TrainingCallback(xgb.callback.TrainingCallback):
    # callback implementation here...
    pass


def train_model(dtrain, dtest, params, num_rounds=1000, early_stopping_rounds=50):
    """
    Train an XGBoost model.
    """
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result
    )
    return model, evals_result


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


def make_prediction(model, dtest):
    """
    Make predictions using the trained XGBoost model.
    """
    predictions = model.predict(dtest)
    return predictions


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


def initialize_wandb():
    """
    Initialize Weights & Biases for experiment tracking.
    """
    if CFG.logging:
        return wandb.init(
            project=CFG.wandb_project_name,
            name=f'{CFG.wandb_run_name}_v{CFG.version}',
            config={
                "n_in": CFG.n_in,
                "n_test": CFG.n_test,
                "params": CFG.params
            },
            job_type='train_model'
        )
    else:
        return None

def train_and_evaluate_model(dtrain, dtest, params):
    """
    Train the XGBoost model and evaluate it.
    """
    callbacks = [WandbCallback()] if CFG.logging else []
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        callbacks=callbacks
    )

    # evaluate the model
    predictions = model.predict(dtest)
    mae = mean_absolute_error(dtest.get_label(), predictions)
    if CFG.logging:
        wandb.log({"Test MAE": mae})

    return model, mae


def save_model(model):
    """
    Save the trained model to a specified path.
    """
    model.save_model(os.path.join(CFG.models_path, 'xgb_model.json'))


def load_model(filepath):
    """
    Load a saved XGBoost model from a file.
    """
    model = xgb.Booster()
    model.load_model(filepath)
    return model


def evaluate_model(model, data, true_labels):
    """
    Evaluate a trained model on a given dataset.
    """
    predictions = model.predict(data)
    mae = mean_absolute_error(true_labels, predictions)
    print(f'Validation MAE: {mae}')
    return mae, predictions