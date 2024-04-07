"""
- scaling is important. do you scale both training and target?
- look ahead is the forecast horizon
- look back: how many time steps to look back
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CFG
from utils import normalize_columns
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm


class LstmCFG:
    n_splits = 5
    n_features = 33
    input_size = 1  # the number of input features in dataset
    hidden_layer_size = 50
    output_size = 1
    learning_rate = 0.00001
    batch_size = 128
    epochs = 10
    sequence_length = 336  # one week of 30-minute sample intervals


class DemandDataset(Dataset):
    def __init__(self, df, label_col, sequence_length=LstmCFG.sequence_length):
        self.df = df
        self.label_col = label_col
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, index):
        # select the sequence of rows, but exclude label column
        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)

        # get the label (target value) for the end of the sequence
        label = self.df.iloc[index + self.sequence_length][self.label_col]

        # convert to tensor
        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float)
        label_tensor = torch.tensor([label], dtype=torch.float)
        # print(f"Sequence tensor shape: {sequence_tensor.shape}, Label tensor shape: {label_tensor.shape}")
        return sequence_tensor, label_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,
                            batch_first=True)  # Ensure LSTM expects batch_size as the first dim
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # No need for manual reshaping if using batch_first=True in LSTM initialization
        # The input_seq is expected to have shape [batch_size, sequence_length, n_features]
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input_seq)
        last_timestep_output = lstm_out[:, -1, :]
        predictions = self.linear(last_timestep_output)
        return predictions


def encode_cyclical_features(df, column, max_value):
    df.loc[:, column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df.loc[:, column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df


# Define the maximum values for each cyclical feature
max_values = {
    'hour': 24,
    'dow': 7,
    'doy': 365,  # or 366 for leap years to be more precise
    'month': 12,
    'quarter': 4
}


if __name__ == "__main__":
    CFG = CFG()

    nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

    nsw_df.drop(columns=['daily_avg_actual', 'daily_avg_forecast'], inplace=True)

    # print(nsw_df.isna().sum())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # define a cutoff date 7 days before the last date in df
    cutoff_date = nsw_df.index.max() - pd.Timedelta(days=7)

    # split the data using cutoff date
    train_df = nsw_df[nsw_df.index <= cutoff_date]
    test_df = nsw_df[nsw_df.index > cutoff_date]

    # Normalize the training data  save scaler
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

    # Input features include both the normalized features and the sine/cosine encoded features
    input_features = ['normalised_total_demand',
                      'normalised_forecast_demand',
                      'normalised_temperature',
                      'normalised_rrp',
                      'normalised_forecast_error',
                      'normalised_smoothed_forecast_demand',
                      'hour_sin',
                      'hour_cos',
                      'dow_sin',
                      'dow_cos',
                      ]

    label_col = 'normalised_total_demand'

    input_size = len(input_features)

    # dataset
    train_dataset = DemandDataset(
        train_df[input_features + [label_col]],
        label_col=label_col,
        sequence_length=LstmCFG.sequence_length
    )

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=LstmCFG.n_splits)

    # Prepare an example feature matrix and target vector (for demonstration)
    # Normally, you'd use your full dataset here, but with LSTMs, handling sequences requires a custom approach
    X = np.array(train_df[input_features])  # This is a placeholder; you'll need to adjust it based on your LSTM data preparation
    y = np.array(train_df[label_col])

    # Assume train_df is prepared and includes both features and the target variable

    for fold, (train_index, val_index) in enumerate(tscv.split(train_df)):
        print(f"Fold {fold + 1}/{LstmCFG.n_splits}")

        # Prepare training and validation sequences
        train_sequences = DemandDataset(
            df=train_df.iloc[train_index],
            label_col=label_col,
            sequence_length=LstmCFG.sequence_length
        )
        val_sequences = DemandDataset(
            df=train_df.iloc[val_index],
            label_col=label_col,
            sequence_length=LstmCFG.sequence_length
        )

        train_loader = DataLoader(
            train_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )

        val_loader = DataLoader(
            val_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )

        # Re-initialize model and optimizer at the start of each fold
        model = LSTMModel(
            LstmCFG.n_features,
            LstmCFG.hidden_layer_size,
            LstmCFG.output_size
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LstmCFG.learning_rate)

        loss_function = nn.L1Loss()

        # Training loop for the current fold
        for epoch in range(LstmCFG.epochs):
            model.train()
            for sequences, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                y_pred = model(sequences)
                # print(f"Predictions shape: {y_pred.shape}, Labels shape: {labels.shape}")
                loss = loss_function(y_pred, labels)
                # if torch.isnan(loss):
                # print("NaN detected in loss")
                loss.backward()
                optimizer.step()
                # for param in model.parameters():
                # if param.grad is not None and torch.isnan(
                    #         param.grad).any():
                # print("NaN detected in gradients")

            # Validation loop for the current fold
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                    sequences, labels = sequences.to(device), labels.to(device)
                    y_pred = model(sequences)
                    val_loss += loss_function(y_pred, labels).item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
