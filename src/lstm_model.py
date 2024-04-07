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
from config import CFG, LstmCFG
from utils import normalize_columns


class DemandDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=100):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        end = min(index + self.sequence_length, len(self.sequences) - 1)
        sequence = self.sequences[index:end]
        label = self.labels[end]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


if __name__ == "__main__":
    CFG = CFG()

    nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

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

    # apply the saved scalers to the test data without fitting
    for original_col, normalized_col in column_mapping.items():
        test_df[normalized_col] = scalers[original_col].transform(test_df[[original_col]])

    # dataset & dataLoader
    train_dataset = DemandDataset(
        dataset_sequences,
        dataset_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=LstmCFG.batch_size,
        shuffle=True
    )

    # model
    model = LSTMModel(LstmCFG.input_size, LstmCFG.hidden_layer_size, LstmCFG.output_size)
    model = model.to(device)

    # loss
    loss_function = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LstmCFG.learning_rate)

    # training loop
    for epoch in range(LstmCFG.epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(sequences)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

"""

OK this is the code as it currently looks. Now i think i need to address your point about normalizing the time features

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CFG, LstmCFG
from utils import normalize_columns


class DemandDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=100):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        end = min(index + self.sequence_length, len(self.sequences) - 1)
        sequence = self.sequences[index:end]
        label = self.labels[end]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


if __name__ == "__main__":
    CFG = CFG()

    nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

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

    # apply the saved scalers to the test data without fitting
    for original_col, normalized_col in column_mapping.items():
        test_df[normalized_col] = scalers[original_col].transform(test_df[[original_col]])
"""