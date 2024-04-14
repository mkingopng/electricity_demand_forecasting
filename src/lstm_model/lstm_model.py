"""

"""
import torch
import torch.nn as nn
import numpy as np


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