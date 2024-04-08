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
from config import CFG
from utils import normalize_columns
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv

load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)


class LstmCFG:
    n_folds = 5
    n_features = 33
    input_size = 1
    hidden_layer_size = 50
    output_size = 1
    lr = 0.0001
    batch_size = 256
    epochs = 10
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    num_layers = 2
    weight_decay = 1e-5


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
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            batch_first=True,
            num_layers=LstmCFG.num_layers,
            bidirectional=True)
        self.dropout = nn.Dropout(LstmCFG.dropout)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        """
        forward pass through the network
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input_seq)
        last_timestep_output = lstm_out[:, -1, :]
        dropped_out = self.dropout(last_timestep_output)
        linear_output = self.linear(dropped_out)
        predictions = self.tanh(linear_output)
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


# define the maximum values for each cyclical feature
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

    wandb_config = {
        "n_folds": LstmCFG.n_folds,
        "n_features": LstmCFG.n_features,
        "hidden layers": LstmCFG.hidden_layer_size,
        "learning_rate": LstmCFG.lr,
        "batch_size": LstmCFG.batch_size,
        "epochs": LstmCFG.epochs,
        "sequence_length": LstmCFG.seq_length,
        "dropout": LstmCFG.dropout,
        "num_layers": LstmCFG.num_layers,
        "weight_decay": LstmCFG.weight_decay
    }

    if CFG.logging:
        wandb.init(
            project=CFG.wandb_project_name,
            name=f'{CFG.wandb_run_name}_v{CFG.version}',
            config=wandb_config,
            job_type='train_model'
        )

    nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))

    nsw_df.drop(
        columns=['daily_avg_actual', 'daily_avg_forecast'],
        inplace=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define a cutoff date 7 days before the last date in df
    cutoff_date = nsw_df.index.max() - pd.Timedelta(days=7)

    # split the data using cutoff date
    train_df = nsw_df[nsw_df.index <= cutoff_date].copy()
    test_df = nsw_df[nsw_df.index > cutoff_date].copy()

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
        test_df.loc[:, normalized_col] = scalers[original_col].transform(test_df[[original_col]])

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

    for fold, (train_index, val_index) in enumerate(tscv.split(train_df)):
        print(f"Fold {fold + 1}/{LstmCFG.n_folds}")
        epoch_train_losses = []
        epoch_test_losses = []

        # prepare train & test sequences
        train_sequences = DemandDataset(
            df=train_df.iloc[train_index],
            label_col=label_col,
            sequence_length=LstmCFG.seq_length
        )
        test_sequences = DemandDataset(
            df=train_df.iloc[val_index],
            label_col=label_col,
            sequence_length=LstmCFG.seq_length
        )
        train_loader = DataLoader(
            train_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )

        # re-initialise model and optimiser at start of each fold
        model = LSTMModel(
            LstmCFG.n_features,
            LstmCFG.hidden_layer_size,
            LstmCFG.output_size
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=LstmCFG.lr,
            weight_decay=LstmCFG.weight_decay
        )

        loss_function = nn.L1Loss()

        # training loop for the current fold
        for epoch in range(LstmCFG.epochs):
            total_train_loss = 0
            total_test_loss = 0
            num_train_batches = 0
            num_test_batches = 0

            model.train()
            for sequences, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                y_pred = model(sequences)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1
            avg_train_loss = total_train_loss / num_train_batches
            if CFG.logging:
                wandb.log({"training loss": avg_train_loss})
            epoch_train_losses.append(avg_train_loss)

            # test loop for the current fold
            model.eval()
            with torch.no_grad():
                for sequences, labels in tqdm(test_loader, desc=f"Test Epoch {epoch + 1}"):
                    sequences, labels = sequences.to(device), labels.to(device)
                    y_pred = model(sequences)
                    total_test_loss += loss_function(y_pred, labels).item()
                    num_test_batches += 1
            avg_test_loss = total_test_loss / num_test_batches
            if CFG.logging:
                wandb.log({"test loss": avg_test_loss})
                wandb.log({"gap": avg_train_loss - avg_test_loss})
            epoch_test_losses.append(avg_test_loss)
            print(f"""
            Epoch {epoch + 1}, 
            Train Loss: {avg_train_loss:.4f}, 
            Test Loss: {avg_test_loss:.4f},
            gap: {avg_train_loss - avg_test_loss:.4f}
            """)

        best_train_loss = min(epoch_train_losses)
        all_folds_train_losses.append(best_train_loss)

        best_test_loss = min(epoch_test_losses)
        all_folds_test_losses.append(best_test_loss)

        print(f"Best Train Loss in fold {fold + 1}: {best_train_loss:.4f}")
        print(f"Best Test Loss in fold {fold + 1}: {best_test_loss:.4f}")

        plot_loss_curves(
            epoch_train_losses,
            epoch_test_losses,
            title=f"Fold {fold + 1} Loss Curves"
        )

    model_train_loss = sum(all_folds_train_losses) / len(all_folds_train_losses)
    if CFG.logging:
        wandb.log({"model training loss": model_train_loss})
    print(f"Model train loss: {model_train_loss:.4f}")

    model_test_loss = sum(all_folds_test_losses) / len(all_folds_test_losses)
    if CFG.logging:
        wandb.log({"model test loss": model_test_loss})
    print(f"Model test loss: {model_test_loss:.4f}")

    model_gap = model_train_loss - model_test_loss
    if CFG.logging:
        wandb.log({"model gap": model_gap})

