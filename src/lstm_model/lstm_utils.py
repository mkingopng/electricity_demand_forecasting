"""

"""
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import wandb
from dotenv import load_dotenv
from lstm_config import LstmCFG


def encode_cyclical_features(df, column, max_value):
    """
    transforms a cyclical feature in a DataFrame to two features using sine and
    cosine transformation to capture the cyclical relationship in a way that
    can be understood by machine learning models
    :param df: df containing the cyclical feature to be encoded
    :param column: name of the column to be transformed
    :param max_value: The maximum value the cyclical feature can take. used to
    normalise the data
    :return: DataFrame with the original column replaced by its sine and cosine
    encoded values
    """
    df.loc[:, column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df.loc[:, column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df


def plot_loss_curves(train_losses, test_losses, title="Loss Curves"):
    """
    plot the training and test loss curves for each epoch
    :param train_losses: list of training losses
    :param test_losses: list of test losses
    :param title: title of the plot
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax.plot(epochs, test_losses, 'ro-', label='Test Loss')
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    if LstmCFG.logging:
        wandb.log({title: wandb.Image(fig)})
    plt.close(fig)


def set_seed(seed_value=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # For CUDA
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_columns(df, column_mapping):
    """
    normalizes specified columns in the DataFrame using Z-score normalization.

    :param df: The DataFrame containing the data.
    :param column_mapping: A dictionary where keys are original column names
    and values are normalized column names.

    Returns:
    - A DataFrame with the normalized columns added.
    - A dictionary of scalers used for each column.
    """
    scalers = {}
    for original_col, normalized_col in column_mapping.items():
        scaler = StandardScaler()
        df[normalized_col] = scaler.fit_transform(df[[original_col]])
        scalers[original_col] = scaler
    return df, scalers
