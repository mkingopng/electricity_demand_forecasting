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
import joblib

load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)


class LstmCFG:
    """
    configuration class for the LSTM model
    """
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'lstm'
    data_path = './../data/NSW'
    logging = False
    train = False
    version = 37  # increment for each training run
    n_folds = 2
    epochs = 2
    n_features = 33
    input_size = 33
    num_layers = 1
    hidden_units = 50
    output_size = 1
    batch_size = 1024
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    weight_decay = 0.00001
    lr = 0.0002
    lrs_step_size = 6
    lrs_gamma = 0.4


class DemandDataset(Dataset):
    def __init__(self, df, label_col, sequence_length=LstmCFG.seq_length, forecast_horizon=336):
        """
        initializes the dataset with the dataframe, label column, sequence
        length, and forecast horizon
        :param df: the dataframe containing the dataset
        :param label_col: the target variable column name
        :param sequence_length: number of time steps in each input sequence
        :param forecast_horizon: number of future time steps to predict
        """
        self.df = df
        self.label_col = label_col
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        """
        returns the total number of samples that can be generated from the df,
        adjusted for the sequence length and forecast horizon
        """
        # ensure non-negative length
        total_length = len(self.df) - self.sequence_length - self.forecast_horizon + 1
        return max(0, total_length)

    def __getitem__(self, index):
        """
        generates a sample from the dataset at the specified index
        :param index: the index of the sample to generate
        :return: a tuple containing the input sequence and the target labels
        """
        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)
        label_start = index + self.sequence_length
        label_end = label_start + self.forecast_horizon
        labels = self.df.iloc[label_start:label_end][self.label_col].values
        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # reshape to [forecast_horizon, 1]
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(100, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        defines the forward pass of the model
        :param x: the input sequence tensor
        :return: the output tensor
        """
        self.lstm.flatten_parameters()
        print(f"Input shape to LSTM: {x.shape}")
        x, _ = self.lstm(x)
        print(f"Output shape from LSTM: {x.shape}")
        x = self.dropout(x)
        x = self.linear(x)
        print(f"Shape after linear layer: {x.shape}")
        x = self.tanh(x)
        print(f"Final output shape: {x.shape}")
        return x


class EarlyStopping:
    """
    early stopping to stop the training when the loss does not improve after
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
        self.test_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.verbose = verbose

    def __call__(self, test_loss, model):
        """
        calls the EarlyStopping instance during training to check if early
        stopping criteria are met
        :param test_loss: the current validation loss
        :param model: the model being trained
        """
        score = -test_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
            self.counter = 0

    def save_checkpoint(self, test_loss, model):
        """
        Saves the current best model if the validation loss decreases
        :param test_loss: the current validation loss
        :param model: the model to be saved
        """
        if self.verbose:
            print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.test_loss_min = test_loss

    def reset(self):
        """
        resets the early stopping counter and best score
        """
        self.counter = 0
        self.best_score = None
        self.test_loss_min = np.Inf


def encode_cyclical_features(df, column, max_value, inplace=True):
    """
    Transforms a cyclical feature in a DataFrame to two features using sine and
    cosine transformations.
    :param df: DataFrame containing the cyclical feature to be encoded.
    :param column: Name of the column to be transformed.
    :param max_value: The maximum value the cyclical feature can take, used to
    normalize the data.
    :param inplace: Whether to modify the DataFrame in place or return a new df
    :return: DataFrame with the original column replaced by its sine and cosine
    encoded values.
    """
    if not inplace:
        df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")
    if max_value == 0:
        raise ValueError("max_value cannot be zero.")
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
    if train_losses:
        ax.plot(epochs, train_losses, 'bo-', label='Training Loss')
    if test_losses:
        test_epochs = range(1, len(test_losses) + 1)
        ax.plot(test_epochs, test_losses, 'ro-', label='Test Loss')
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    if CFG.logging:
        wandb.log({title: wandb.Image(fig)})
    plt.close(fig)


def set_seed(seed_value=42):
    """
    set seed for reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
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
    ##########################################################################
    # load data
    ##########################################################################
    nsw_df = pd.read_parquet(os.path.join(LstmCFG.data_path, 'nsw_df.parquet'))
    nsw_df.drop(
        columns=['daily_avg_actual', 'daily_avg_forecast'],
        inplace=True
    )
    # print(f'nsw_df columns: {nsw_df.shape[1]}')  # number of columns in df

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
    # split dataset into train, test, and validation sets
##############################################################################
    # define a cutoff date from the last date in df
    cutoff_date1 = nsw_df.index.max() - pd.Timedelta(days=14)
    cutoff_date2 = nsw_df.index.max() - pd.Timedelta(days=28)

    # split the data using cutoff date
    train_df = nsw_df[nsw_df.index <= cutoff_date2].copy()
    # print(f'train_df columns: {train_df.shape[1]}')  # dubugging line

    test_df = nsw_df[(nsw_df.index > cutoff_date2) & (nsw_df.index <= cutoff_date1)].copy()
    # print(f'nsw_df columns: {test_df.shape[1]}')  # dubugging line

    val_df = nsw_df[nsw_df.index > cutoff_date1].copy()
    # print(f'val_df columns: {val_df.shape[1]}')  # dubugging line

##############################################################################
    # normalize the training data, save scaler
##############################################################################
    train_df, scalers = normalize_columns(
        train_df,
        column_mapping
    )
    print(f'train_df shape: {train_df.shape}')

    val_df, scalers = normalize_columns(
        val_df,
        column_mapping
    )
    print(f'val_df shape: {val_df.shape}')

##############################################################################
    # encode cyclical features for train, test and val df
##############################################################################
    for col, max_val in max_values.items():
        train_df = encode_cyclical_features(train_df, col, max_val)
        # print(f'encoded train_df columns: {test_df.shape[1]}')  # dubugging line

        test_df = encode_cyclical_features(test_df, col, max_val)
        # print(f'encoded test_df columns: {test_df.shape[1]}')  # dubugging line

        val_df = encode_cyclical_features(val_df, col, max_val)
        # print(f'encoded val_df columns: {val_df.shape[1]}')  # dubugging line

##############################################################################
    # normalize the test data
    # apply the saved scalers to the test data without fitting
    for original_col, normalized_col in column_mapping.items():
        train_df.loc[:, normalized_col] = scalers[original_col].transform(
            train_df[[original_col]])

        test_df.loc[:, normalized_col] = scalers[original_col].transform(
            test_df[[original_col]])

        val_df.loc[:, normalized_col] = scalers[original_col].transform(
            val_df[[original_col]])

    label_col = 'normalised_total_demand'

    input_size = len(input_features)
    print(f'input size: {input_size}')

    train_dataset = DemandDataset(
        train_df[input_features + [label_col]],
        label_col=label_col,
        sequence_length=LstmCFG.seq_length
    )

    val_dataset = DemandDataset(
        val_df,
        label_col='TOTALDEMAND',
        sequence_length=LstmCFG.seq_length,
        forecast_horizon=336
    )

    set_seed(seed_value=42)

    if LstmCFG.train:
        CFG = LstmCFG()

        wandb_config = {
            "n_folds": LstmCFG.n_folds,
            "n_features": LstmCFG.n_features,
            "hidden layers": LstmCFG.hidden_units,
            "learning_rate": LstmCFG.lr,
            "batch_size": LstmCFG.batch_size,
            "epochs": LstmCFG.epochs,
            "sequence_length": LstmCFG.seq_length,
            "dropout": LstmCFG.dropout,
            "num_layers": LstmCFG.num_layers,
            "weight_decay": LstmCFG.weight_decay,
            "lrs_step_size": LstmCFG.lrs_step_size,
            "lrs_gamma": LstmCFG.lrs_gamma,
        }

        if CFG.logging:
            wandb.init(
                project=CFG.wandb_project_name,
                name=f'{CFG.wandb_run_name}_v{CFG.version}',
                config=wandb_config,
                job_type='train_model'
            )

###############################################################################
        # encode cyclical features for train and test df
###############################################################################


        # time series split
        tscv = TimeSeriesSplit(n_splits=LstmCFG.n_folds)
        X = np.array(train_df[input_features])
        y = np.array(train_df[label_col])
        all_folds_test_losses = []
        all_folds_train_losses = []

##############################################################################
        # loop through each fold
##############################################################################
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

            # train loader
            train_loader = DataLoader(
                train_sequences,
                batch_size=LstmCFG.batch_size,
                shuffle=False
            )
            # print(f"train loader: {len(train_loader)}")  # dubugging line

            # test loader
            test_loader = DataLoader(
                test_sequences,
                batch_size=LstmCFG.batch_size,
                shuffle=False
            )
            # print(f"test loader: {len(test_loader)}")  # dubugging line

            # re-initialise model and optimiser at start of each fold
            model = LSTMModel(
                input_size=LstmCFG.input_size,
                hidden_layer_size=LstmCFG.hidden_units,
                output_size=LstmCFG.output_size,
                dropout=LstmCFG.dropout,
                num_layers=LstmCFG.num_layers
            ).to(device)

            # optimiser
            optimizer = optim.Adam(
                model.parameters(),
                lr=LstmCFG.lr,
                weight_decay=LstmCFG.weight_decay
            )

            # learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=LstmCFG.lrs_step_size,
                gamma=LstmCFG.lrs_gamma
            )

            # loss function
            loss_function = nn.L1Loss()

            # early stopping
            early_stopping = EarlyStopping(
                patience=10,
                verbose=True,
                path='model_checkpoint.pth'
            )

##############################################################################
            # training loop for the current fold
##############################################################################
            for epoch in range(LstmCFG.epochs):
                training_loss = 0
                num_batches = 0
                model.train()

                for sequences, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                    sequences, labels = sequences.to(device), labels.to(device)
                    optimizer.zero_grad()
                    # print(f"Input sequence shape: {sequences.shape}")  # dubugging line
                    predictions = model(sequences)
                    loss = loss_function(predictions, labels)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()
                    num_batches += 1

                # calculate average training loss
                avg_train_loss = training_loss / num_batches

                # append average training loss to list
                epoch_train_losses.append(avg_train_loss)

##############################################################################
                # test loop
##############################################################################
                model.eval()
                total_test_loss = 0
                with torch.no_grad():
                    for sequences, labels in tqdm(test_loader, desc=f"Test Epoch {epoch + 1}"):
                        sequences, labels = sequences.to(device), labels.to(device)
                        predictions = model(sequences)
                        # print(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
                        test_loss = loss_function(predictions, labels)
                        total_test_loss += test_loss.item()

                # early Stopping and logging
                avg_test_loss = total_test_loss / len(test_loader)
                epoch_test_losses.append(avg_test_loss)
                early_stopping(avg_test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

                # log losses for this epoch to Weights & Biases
                if CFG.logging:
                    wandb.log({
                        "epoch": epoch,
                        "training loss": training_loss / len(train_loader),
                        "test loss": avg_test_loss
                    })

                epoch_test_losses.append(avg_test_loss)
                print(f"""
                Epoch {epoch + 1},
                Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f},
                Train MAE: {training_loss:.4f}, 
                Test MAE: {avg_test_loss:.4f},
                gap: {training_loss - avg_test_loss:.4f}
                """)

                early_stopping(avg_test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    torch.save(
                        model.state_dict(),
                        'model_checkpoint.pth')
                    break
            model.load_state_dict(torch.load('model_checkpoint.pth'))
            artifact = wandb.Artifact('model_artifact', type='model')
            artifact.add_file('model_checkpoint.pth')
            if CFG.logging:
                wandb.save('model_checkpoint.pth')

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
        # Save scalers after training
        joblib.dump(scalers, 'scalers.pkl')
        gc.collect()
        torch.cuda.empty_cache()

    # todo: wandb sweeps to optimise hyperparameters
    # todo: inference, ensemble

###############################################################################
    # Inference
    else:
        CFG = LstmCFG()

        # initialize the model
        model = LSTMModel(
            input_size=LstmCFG.input_size,  # fix_me
            hidden_layer_size=LstmCFG.hidden_units,
            output_size=LstmCFG.output_size,
            dropout=LstmCFG.dropout,
            num_layers=LstmCFG.num_layers
        )
        model.load_state_dict(torch.load('model_checkpoint.pth'))
        print(os.path.exists(
            'model_checkpoint.pth'))  # Should return True if the file exists
        print("Size of checkpoint file:",
              os.path.getsize('model_checkpoint.pth'), "bytes")
        model = model.to(device)
        model.eval()

        print("Number of validation samples available:", len(val_dataset))

        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=LstmCFG.batch_size,
                shuffle=False
            )
            predictions = []
            actuals = []
        # if len(val_df) < (LstmCFG.seq_length + 336):
        #     print("Not enough data points for a full sequence and forecast.")
        # else:
        #     scalers = joblib.load('scalers.pkl')
            with torch.no_grad():
                # dummy_input = torch.randn(1, 336, 33).to(
                #     device)  # Random data with the shape of your input
                # model.eval()
                # output = model(dummy_input)
                # print("Dummy Output shape:", output.shape)

                for sequences, labels in val_loader:
                    print(f"Sequences shape: {sequences.shape}, Labels shape: {labels.shape}")
                    sequences, labels = sequences.to(device), labels.to(device)
                    output = model(sequences)  # fix_me
                    predictions.extend(output.cpu().numpy())
                    actuals.extend(labels.cpu().numpy())

                if predictions:
                    predictions = scalers['TOTALDEMAND'].inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                    actuals = scalers['TOTALDEMAND'].inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
                    mae = np.mean(np.abs(predictions - actuals))
                    print(f"Mean Absolute Error (MAE): {mae:.2f}")
                    plt.figure(figsize=(10, 5))
                    plt.plot(predictions, label='Predicted Demand')
                    plt.plot(actuals, label='Actual Demand')
                    plt.title('Comparison of Predicted and Actual Electricity Demand')
                    plt.xlabel('Time')
                    plt.ylabel('Electricity Demand')
                    plt.legend()
                    plt.show()

                else:
                    print("No predictions were made.")
