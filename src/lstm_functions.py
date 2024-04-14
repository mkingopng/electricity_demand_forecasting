"""

"""
import gc
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from utils import normalize_columns
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
from lstm_config import LstmCFG


class DemandDataset(Dataset):
    def __init__(self, df, label_col, sequence_length=LstmCFG.seq_length):
        """
        Initializes the dataset with the dataframe, label column, and sequence
        length
        :param df: The dataframe containing the dataset
        :param label_col: The target variable
        :param sequence_length:
        :return:
        """
        self.df = df
        self.label_col = label_col
        self.sequence_length = sequence_length

    def __len__(self):
        """
        returns the total number of samples that can be generated from the
        dataframe
        :return: the total number of samples
        """
        return len(self.df) - self.sequence_length

    def __getitem__(self, index):
        """
        generates a sample from the dataset at the specified index
        :param index: the index of the sample to generate
        :return: (tuple) a tuple containing the sequence tensor and the label
        tensor
        """
        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)
        label = self.df.iloc[index + self.sequence_length][self.label_col]
        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float)
        label_tensor = torch.tensor([label], dtype=torch.float)
        return sequence_tensor, label_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout, num_layers):
        """
        initialises an LSTM model with specified architecture parameters
        :param input_size: the number of input features per time step
        :param hidden_layer_size: the number of hidden units in the LSTM layer
        :param output_size: the number of output vectors
        :param dropout: the dropout rate for dropout layers to prevent
        over-fitting
        :param num_layers: the number of layers in the LSTM
        """
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        """
        defines the forward pass of the model
        :param input_seq: (Tensor) The input sequence to the LSTM model
        :return: Tensor: the predictions made by the model
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input_seq)
        # print(f"LSTM output shape: {lstm_out.shape}")  # Debugging line
        last_timestep_output = lstm_out[:, -1, :]
        # print(f"Last timestep output shape: {last_timestep_output.shape}") # Debugging line
        dropped_out = self.dropout(last_timestep_output)
        linear_output = self.linear(dropped_out)
        predictions = self.tanh(linear_output)
        # print(f"Predictions shape: {predictions.shape}") # Debugging line
        return predictions


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    """
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', verbose=False):
        """
        initializes the EarlyStopping mechanic with custom configuration
        :param patience: (int) number of epochs with no improvement after which
        training will be stopped. default: 5.
        :param delta: (float) Minimum change in monitored quantity to qualify
        as an improvement. default: 0.
        :param path: (str) Path for the checkpoint to be saved to.
        default: 'checkpoint.pt'
        :param verbose: (bool) If True, prints a message for each validation
        loss improvement. Default: False
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model):
        """
        calls the EarlyStopping instance during training to check if early
        stopping criteria are met
        :param val_loss: the current validation loss
        :param model: the model being trained
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the current best model if the validation loss decreases
        :param val_loss: the current validation loss
        :param model: the model to be saved
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
    if CFG.logging:
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