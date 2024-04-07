import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import CFG
from utils import normalize_columns
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import matplotlib.pyplot as plt


class LstmCFG:
    n_folds = 5
    n_features = 33
    input_size = 1
    hidden_layer_size = 50
    output_size = 1
    lr = 0.0001
    batch_size = 256
    epochs = 5
    seq_length = 336  # 336 one week of 30-minute sample intervals


class DemandDataset(Dataset):
    def __init__(self, df, label_col, sequence_length=LstmCFG.seq_length):
        self.df = df
        self.label_col = label_col
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, index):
        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)
        label = self.df.iloc[index + self.sequence_length][self.label_col]
        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float)
        label_tensor = torch.tensor([label], dtype=torch.float)
        return sequence_tensor, label_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        forward pass through the network
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input_seq)
        last_timestep_output = lstm_out[:, -1, :]
        predictions = self.linear(last_timestep_output)
        return predictions


def encode_cyclical_features(df, column, max_value):
    """
    encode a cyclical feature using sine and cosine transformations
    """
    df.loc[:, column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df.loc[:, column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df


def plot_loss_curves(train_losses, test_losses, title="Loss Curves"):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def find_learning_rate(
        model, data_loader, criterion, init_lr=1e-7, final_lr=10., beta=0.98
        ):
    num = len(data_loader) - 1
    mult = (final_lr / init_lr) ** (1 / num)
    lr = init_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    avg_loss = 0.
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in data_loader:
        batch_num += 1
        optimizer.param_groups[0]['lr'] = lr
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(lr)

        # Do the optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the lr for the next step
        lr *= mult

    plt.plot([np.log10(x) for x in log_lrs], losses)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.show()


# Define the maximum values for each cyclical feature
max_values = {
    'hour': 24,
    'dow': 7,
    'doy': 365,  # or 366 for leap years to be more precise
    'month': 12,
    'quarter': 4
}

column_mapping = {
        'TOTALDEMAND': 'normalised_total_demand',
        'FORECASTDEMAND': 'normalised_forecast_demand',
        'TEMPERATURE': 'normalised_temperature',
        'rrp': 'normalised_rrp',
        'forecast_error': 'normalised_forecast_error',
        'smoothed_forecast_demand': 'normalised_smoothed_forecast_demand',
        'smoothed_total_demand': 'normalised_smoothed_total_demand',
        'smoothed_temperature': 'normalised_smoothed_temperature',
    }

input_features = [
        'normalised_total_demand',
        'normalised_forecast_demand',
        'normalised_temperature',
        'normalised_rrp',
        'normalised_forecast_error',
        'normalised_smoothed_forecast_demand',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos'
    ]

if __name__ == "__main__":
    CFG = CFG()

    nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

    nsw_df.drop(
        columns=['daily_avg_actual', 'daily_avg_forecast'],
        inplace=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define a cutoff date 7 days before the last date in df
    cutoff_date = nsw_df.index.max() - pd.Timedelta(days=7)

    # split the data using cutoff date
    train_df = nsw_df[nsw_df.index <= cutoff_date]
    test_df = nsw_df[nsw_df.index > cutoff_date]

    # normalize the training data, save scaler
    train_df, scalers = normalize_columns(
        train_df,
        column_mapping
    )

    # encode cyclical features for both training and test data
    for col, max_val in max_values.items():
        train_df = encode_cyclical_features(train_df, col, max_val)
        test_df = encode_cyclical_features(test_df, col, max_val)

    # apply the saved scalers to the test data without fitting
    for original_col, normalized_col in column_mapping.items():
        test_df[normalized_col] = scalers[original_col].transform(test_df[[original_col]])

    label_col = 'normalised_total_demand'
    input_size = len(input_features)

    train_dataset = DemandDataset(
        train_df[input_features + [label_col]],
        label_col=label_col,
        sequence_length=LstmCFG.seq_length
    )

    tscv = TimeSeriesSplit(n_splits=LstmCFG.n_folds)
    X = np.array(train_df[input_features])
    y = np.array(train_df[label_col])
    all_folds_test_losses = []
    all_folds_train_losses = []

    sample_dataset = DemandDataset(
        df=train_df[input_features + [label_col]],
        label_col=label_col,
        sequence_length=LstmCFG.seq_length
    )
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=LstmCFG.batch_size,
        shuffle=False  # Shuffle to get a random representative sample
    )

    model = LSTMModel(
        LstmCFG.n_features,
        LstmCFG.hidden_layer_size,
        LstmCFG.output_size
    ).to(device)

    criterion = nn.L1Loss()

    find_learning_rate(model, sample_loader, criterion)
