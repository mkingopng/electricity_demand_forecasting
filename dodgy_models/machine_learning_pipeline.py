"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb


# config class
class CFG:
    def __init__(self):
        self.n_splits = 5
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10
        self.data_path = '../data'
        self.img_dim1 = 20
        self.img_dim2 = 10
        self.wandb_project = 'electricity_demand_forecasting'  # set project name
        self.wandb_run_name = 'simple_exponential_smoothing'  # name this run


CFG = CFG()

wandb.init(
    project=CFG.wandb_project,
    config=CFG.__dict__
)


def load_data():
    """
    load data
    :return:
    """
    dates = pd.date_range(
        start="2020-01-01",
        end="2020-12-31",
        freq="30min"
    )
    data = np.random.randn(len(dates), 1)
    df = pd.DataFrame(
        data,
        columns=["value"],
        index=dates
    )
    return df


def preprocess_data(df):
    # implement preprocessing and feature engineering here
    # I think we've already done this elsewhere
    # for now this is a placeholder
    return df


def split_data(df):
    """
    split data into train, test, and validation sets, using the first 80% for
    training, next 10% for validation, and last 10% for testing
    :param df:
    :return:
    """
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.9)
    train_data = df[:train_size]
    val_data = df[train_size:val_size]
    test_data = df[val_size:]
    return train_data, val_data, test_data


def create_data_loader(X, y, batch_size):
    """
    Converts X, y DataFrames into a DataLoader.
    """
    dataset = TensorDataset(
        torch.tensor(X.values, dtype=torch.float),
        torch.tensor(y.values, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def time_series_cv_splits(df, n_splits, batch_size):
    """
    Prepare data for time series cross-validation
    This version assumes 'df' includes both features and target.
    Adjust 'features' and 'target' based on your actual DataFrame structure.
    :param df: DataFrame containing the time series data
    :param n_splits: Number of splits for cross-validation
    :return: A generator that yields train/test splits
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(df):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        # Assuming the last column is the target
        y_train, y_test = X_train.iloc[:, -1], X_test.iloc[:, -1]
        X_train, X_test = X_train.iloc[:, :-1], X_test.iloc[:, :-1]

        # further split X_train for validation
        # eg: use the last 20% of X_train for validation
        val_size = int(len(X_train) * 0.8)
        X_val, y_val = X_train[val_size:], y_train[val_size:]
        X_train, y_train = X_train[:val_size], y_train[:val_size]

        train_loader = create_data_loader(X_train, y_train, batch_size)
        val_loader = create_data_loader(X_val, y_val, batch_size)
        test_loader = create_data_loader(X_test, y_test, batch_size)

        yield train_loader, val_loader, test_loader


# define torch models
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # define neural network architecture here
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def train_model(model, train_loader, val_loader, config):
    # implement training loop here
    # use train_loader for training steps
    # use val_loader for validation steps
    pass


def evaluate_model(model, test_loader):
    # implement model evaluation here
    pass


if __name__ == "__main__":
    df = load_data()
    df_processed = preprocess_data(df)

    # convert your data into a format compatible with each model
    # for example, for PyTorch, convert data into DataLoader objects

    model = PyTorchModel()  # example only
    # implement specific model training and evaluation logic here
    # this could involve statsmodels, scikit-learn models, or PyTorch model

    for fold, (train_loader, val_loader, test_loader) in enumerate(
            time_series_cv_splits(df_processed, CFG.n_splits, CFG.batch_size)):
        # train and validate your model on the current fold
        train_model(model, train_loader, val_loader, CFG)

        # evaluate your model on the test set of the current fold
        evaluate_model(model, test_loader)

        # Log fold results, adj model, perform necessary actions between folds
