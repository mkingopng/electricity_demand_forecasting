"""
note: this model runs fine on cpu. unlike unstructured data, this kind of
problem doesn't get a huge benefit from using GPU

lstm model
- 2x LSTM layers
- bidirectional LSTM
- 50 hidden units
- Tanh activation function
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
    input_size = 33
    hidden_layer_size = 50
    output_size = 1
    lr = 0.0001
    batch_size = 256
    epochs = 10
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    num_layers = 2
    weight_decay = 1e-5
    lrs_step_size = 3
    lrs_gamma = 0.05


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
    def __init__(self, input_size, hidden_layer_size, output_size, dropout, num_layers):
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
        forward pass through the network
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input_seq)
        # print(f"LSTM output shape: {lstm_out.shape}")  # Debugging line
        last_timestep_output = lstm_out[:, -1, :]
        # print(f"Last timestep output shape: {last_timestep_output.shape}")  # Debugging line
        dropped_out = self.dropout(last_timestep_output)
        linear_output = self.linear(dropped_out)
        predictions = self.tanh(linear_output)
        # print(f"Predictions shape: {predictions.shape}")  # Debugging line
        return predictions


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
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
        """Saves model when validation loss decrease"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def encode_cyclical_features(df, column, max_value):
    """
    encode a cyclical feature using sine and cosine transformations
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
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # For CUDA
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# define the maximum values for each cyclical feature
max_values = {
    'hour': 24,
    'dow': 7,
    'doy': 365,  # todo: 366 for leap years to be more precise
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
    set_seed(seed_value=42)

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
        "weight_decay": LstmCFG.weight_decay,
        "lrs_step_size": LstmCFG.lrs_step_size,
        "lrs_gamma": LstmCFG.lrs_gamma
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
    # print(f'nsw_df columns: {nsw_df.shape[1]}')  # number of columns in df

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define a cutoff date from the last date in df
    cutoff_date = nsw_df.index.max() - pd.Timedelta(days=7)

    # split the data using cutoff date
    train_df = nsw_df[nsw_df.index <= cutoff_date].copy()
    # print(f'train_df columns: {train_df.shape[1]}')  # dubugging line

    test_df = nsw_df[nsw_df.index > cutoff_date].copy()
    # print(f'nsw_df columns: {test_df.shape[1]}')  # dubugging line

    # normalize the training data, save scaler
    train_df, scalers = normalize_columns(
        train_df,
        column_mapping
    )

    # encode cyclical features for both training and test data
    for col, max_val in max_values.items():
        train_df = encode_cyclical_features(train_df, col, max_val)
        # print(f'encoded train_df columns: {test_df.shape[1]}')
        test_df = encode_cyclical_features(test_df, col, max_val)
        # print(f'encoded test_df columns: {test_df.shape[1]}')

    # apply the saved scalers to the test data without fitting
    for original_col, normalized_col in column_mapping.items():
        test_df.loc[:, normalized_col] = scalers[original_col].transform(test_df[[original_col]])

    label_col = 'normalised_total_demand'

    input_size = len(input_features)
    # print(f'input size: {input_size}')

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
        # print(f"Fold {fold + 1}/{LstmCFG.n_folds}")  # dubugging line
        epoch_train_losses = []
        epoch_test_losses = []

        # prepare train & test sequences
        train_sequences = DemandDataset(
            df=train_df.iloc[train_index],
            label_col=label_col,
            sequence_length=LstmCFG.seq_length
        )
        # print(f"train sequences: {len(train_sequences)}")  # dubugging line

        test_sequences = DemandDataset(
            df=train_df.iloc[val_index],
            label_col=label_col,
            sequence_length=LstmCFG.seq_length
        )
        # print(f"test sequences: {len(test_sequences)}")  # dubugging line

        train_loader = DataLoader(
            train_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )
        # print(f"train loader: {len(train_loader)}")  # dubugging line

        test_loader = DataLoader(
            test_sequences,
            batch_size=LstmCFG.batch_size,
            shuffle=False
        )
        # print(f"test loader: {len(test_loader)}")  # dubugging line

        # re-initialise model and optimiser at start of each fold
        model = LSTMModel(
            input_size=LstmCFG.input_size,
            hidden_layer_size=LstmCFG.hidden_layer_size,
            output_size=LstmCFG.output_size,
            dropout=LstmCFG.dropout,
            num_layers=LstmCFG.num_layers
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=LstmCFG.lr,
            weight_decay=LstmCFG.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LstmCFG.lrs_step_size,
            gamma=LstmCFG.lrs_gamma
        )

        loss_function = nn.L1Loss()

        early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path='model_checkpoint.pt'
        )

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
                # print(f"Input sequence shape: {sequences.shape}")  # dubugging line
                y_pred = model(sequences)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1
            lr_scheduler.step()
            if CFG.logging:
                wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})

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
                wandb.log(
                    {
                        "test loss": avg_test_loss,
                        "gap": avg_train_loss - avg_test_loss
                     }
                )
            epoch_test_losses.append(avg_test_loss)
            print(f"""
            Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f},
            Epoch {epoch + 1}, 
            Train Loss: {avg_train_loss:.4f}, 
            Test Loss: {avg_test_loss:.4f},
            gap: {avg_train_loss - avg_test_loss:.4f}
            """)

            early_stopping(avg_test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                torch.save(model.state_dict(), 'model_checkpoint.pt')
                break
        model.load_state_dict(torch.load('model_checkpoint.pt'))
        artifact = wandb.Artifact('model_artifact', type='model')
        artifact.add_file('model_checkpoint.pt')
        if CFG.logging:
            wandb.save('model_checkpoint.pt')

        best_train_loss = min(epoch_train_losses)
        all_folds_train_losses.append(best_train_loss)

        best_test_loss = min(epoch_test_losses)
        all_folds_test_losses.append(best_test_loss)

        print(f"Best Train Loss in fold {fold + 1}: {best_train_loss:.4f}")
        print(f"Best Test Loss in fold {fold + 1}: {best_test_loss:.4f}")

        plot_loss_curves(
            epoch_train_losses,
            epoch_test_losses,
            title=f"Fold {fold} Loss Curves"
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
        wandb.finish()

    gc.collect()
    torch.cuda.empty_cache()

    # inference

    # # initialize the model
    # model = LSTMModel(
    #     input_size=LstmCFG.input_size,
    #     hidden_layer_size=LstmCFG.hidden_layer_size,
    #     output_size=LstmCFG.output_size,
    #     dropout=LstmCFG.dropout,
    #     num_layers=LstmCFG.num_layers
    # )
    #
    # # load the state dictionary
    # model.load_state_dict(torch.load('model_final.pt'))
    #
    # # call model.eval() to set dropout and batch normalization layers to
    # # evaluation mode before running inference.
    # model.eval()
    #
    # # Now you can use model for inference
