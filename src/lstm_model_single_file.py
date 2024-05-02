"""

"""
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
from sklearn.preprocessing import MinMaxScaler
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
    data_path = '../data/NSW'
    images_path = '../images'
    model_path = '../trained_models'
    version = 1  # increment for each training run
    model_name = f'lstm_trained_model_v{version}'
    logging = True
    train = True
    seed = 42
    n_folds = 5
    epochs = 5
    # n_features = 97
    input_size = 98
    num_layers = 2
    hidden_units = 1
    output_size = 1
    batch_size = 1024
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    weight_decay = 0.00001
    lr = 0.0002
    lrs_step_size = 6
    lrs_gamma = 0.4
    patience = 10


# define the maximum values for each cyclical feature
max_values = {
    'hour': 24,
    'dow': 7,
    'doy': 365,  # todo: 366 for leap years to be more precise
    'month': 12,
    'quarter': 4,
    'week_of_year': 52,
    'minutes_past_midnight': 1439
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

continuous_features = [
        'TOTALDEMAND', 'FORECASTDEMAND', 'TEMPERATURE', 'rrp',
        'daily_avg_actual', 'daily_avg_forecast', 'forecast_error',
        'smoothed_forecast_demand', 'smoothed_total_demand',
        'smoothed_temperature'
    ]

cyclical_features = ['hour', 'dow', 'doy', 'month', 'quarter',
                     'week_of_year', 'minutes_past_midnight']

categorical_features = ['day_of_month', 'is_weekend', 'part_of_day',
                        'season', 'is_business_day', 'season_name', 'year']

wandb_config = {
            "n_folds": LstmCFG.n_folds,
            "n_features": LstmCFG.input_size,
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

def preprocess_data(path, cutoff_days_test=14, cutoff_days_val=28):
    """
    preprocess data for LSTM training and validation
    :param path: (str) path to the data file
    :param cutoff_days_test: (int) days to offset the test split
    :param cutoff_days_val: (int) days to offset the validation split
    :returns tuple: train_dataset, test_dataset, val_dataset
    """
    nsw_df = pd.read_parquet(path)
    nsw_df.dropna(inplace=True)

    # apply cyclical encoding first
    for col in cyclical_features:
        nsw_df = encode_cyclical_features(nsw_df, col, max_values[col])

    # apply one-hot encoding second
    nsw_df = pd.get_dummies(nsw_df, columns=categorical_features)

    # define cutoff dates
    cutoff_date1 = nsw_df.index.max() - pd.Timedelta(days=cutoff_days_test)
    cutoff_date2 = nsw_df.index.max() - pd.Timedelta(days=cutoff_days_val)

    train_df = nsw_df[nsw_df.index <= cutoff_date2].copy()
    # print(train_df.dtypes)
    test_df = nsw_df[(nsw_df.index > cutoff_date2) & (nsw_df.index <= cutoff_date1)].copy()
    val_df = nsw_df[nsw_df.index > cutoff_date1].copy()

    # normalise continuous features using MinMax
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df['TOTALDEMAND_scaled'] = scaler.fit_transform(train_df[['TOTALDEMAND']])
    test_df['TOTALDEMAND_scaled'] = scaler.transform(test_df[['TOTALDEMAND']])
    val_df['TOTALDEMAND_scaled'] = scaler.transform(val_df[['TOTALDEMAND']])

    # train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
    # test_df[continuous_features] = scaler.transform(test_df[continuous_features])
    # val_df[continuous_features] = scaler.transform(val_df[continuous_features])
    # print("Number of features after one-hot encoding:", len(train_df.columns))  # debugging line

    # additional normalization  CHECK THIS
    train_df, _ = normalize_columns(train_df, column_mapping)
    test_df, _ = normalize_columns(test_df, column_mapping)
    val_df, _ = normalize_columns(val_df, column_mapping)

    # print(train_df.columns)
    # print("Feature range after scaling (min, max):",
    #       train_df['TOTALDEMAND'].min(),
    #       train_df['TOTALDEMAND'].max())

    # create train dataset
    train_dataset = DemandDataset(
        train_df,
        'TOTALDEMAND',
        sequence_length=LstmCFG.seq_length
    )

    # test dataset
    test_dataset = DemandDataset(
        test_df,
        'TOTALDEMAND',
        sequence_length=LstmCFG.seq_length
    )

    # validation dataset
    val_dataset = DemandDataset(
        val_df,
        'TOTALDEMAND',
        sequence_length=LstmCFG.seq_length,
        forecast_horizon=336
    )
    return train_dataset, test_dataset, val_dataset, scaler


def encode_day_of_month(df, date_column='date'):
    df['day_of_month_sin'] = np.sin(2 * np.pi * df[date_column].dt.day / df[date_column].dt.days_in_month)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df[date_column].dt.day / df[date_column].dt.days_in_month)
    return df


def encode_cyclical_features(df, column, max_value, inplace=True):
    """
    transforms a cyclical feature in a DataFrame to two features using sine and
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
        if index + self.sequence_length >= len(self.df) or index + self.sequence_length + self.forecast_horizon > len(self.df):
            raise IndexError("Index range out of bounds for sequence or labels generation.")

        sequence = self.df.iloc[index:index + self.sequence_length].drop(self.label_col, axis=1)
        if sequence.empty:
            raise ValueError("Sequence slice resulted in an empty DataFrame.")
        # print("Sequence values:", sequence.values)

        # Handle potential floating-point precision issues and convert to float explicitly
        sequence = sequence.astype(np.float32)

        label_start = index + self.sequence_length
        label_end = label_start + self.forecast_horizon
        labels = self.df.iloc[label_start:label_end][self.label_col].values
        if labels.size == 0:
            raise ValueError("Label slice resulted in an empty array.")

        sequence_tensor = torch.tensor(sequence.values, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # reshape to [forecast_horizon, 1]

        # Debug print statements to check tensor shapes
        # print(f"Sequence index: {index}")
        # print(f"Sequence shape: {sequence_tensor.shape}, Label shape: {label_tensor.shape}")
        # print(f"First few entries of sequence tensor: {sequence_tensor[:5]}")
        # print(f"First few entries of label tensor: {label_tensor[:5]}")

        return sequence_tensor, label_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, seq_length, dropout, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),  # dropout is used if num_layers > 1
            batch_first=True,
            bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        # apply linear layer to each time step output from LSTM
        self.linear = nn.Linear(hidden_layer_size, seq_length)

    def forward(self, x):
        # print(f"Input shape to LSTM: {x.shape}")
        x, (h_n, c_n) = self.lstm(x)
        # print(f"Output shape from LSTM: {x.shape}")  # should be [batch, seq_len, num_directions * hidden_size]
        x = self.dropout(x)
        x = x.contiguous().view(-1, x.shape[2])
        x = self.linear(x)
        # Reshape back to [batch, seq_len, output_size]
        x = x.view(-1, 336, 1)  # Assuming output_size is 1
        # print(f"Linear Output Stats - Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}")  # debugging line
        # print(f"Shape after linear layer: {x.shape}")  # debugging line
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
        saves the current best model if the validation loss decreases
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


def train_model(train_df, test_df, input_features, label_col, CFG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    set_seed(CFG.seed)  # ensure reproducibility

    # call the preprocess function and obtain the scaler
    train_df, test_df, val_df, scaler = preprocess_data(os.path.join(CFG.data_path, 'nsw_df.parquet'))

    # save the scaler
    joblib.dump(scaler, os.path.join(CFG.model_path, 'scaler.pkl'))

    if CFG.train:
        # initialize W&B if logging is enabled
        if CFG.logging:
            wandb.init(
                project=CFG.wandb_project_name,
                name=f'{CFG.wandb_run_name}_v{CFG.version}',
                config=wandb_config,
                job_type='train_model'
            )

        all_folds_test_losses = []
        all_folds_train_losses = []
        # avg_test_loss = []
        # training_loss = []

        # prepare data using Time Series Split
        tscv = TimeSeriesSplit(n_splits=CFG.n_folds)
        fold = 0
        for train_index, test_index in tscv.split(train_df):
            # fold += 1
            print(f"Fold {fold + 1}/{CFG.n_folds}")

            # prepare train and test datasets / data loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                shuffle=True,
                drop_last=True
            )

            # test data loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=CFG.batch_size,
                shuffle=False
            )

            # initialize the LSTM model
            model = LSTMModel(
                input_size=CFG.input_size,
                hidden_layer_size=CFG.hidden_units,
                seq_length=CFG.output_size,
                num_layers=CFG.num_layers,
                dropout=CFG.dropout
            ).to(device)

            # initialise optimiser
            optimizer = optim.Adam(
                model.parameters(),
                lr=CFG.lr,
                weight_decay=CFG.weight_decay
            )

            # initialise lr_scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=CFG.lrs_step_size,
                gamma=CFG.lrs_gamma
            )

            # loss function MSE
            loss_function = nn.L1Loss()

            # initialise early stopping
            early_stopping = EarlyStopping(
                patience=CFG.patience,
                verbose=True,
                path=f'{CFG.model_path}/{CFG.model_name}.pth'
            )

            # training loop for the current fold
            for epoch in range(CFG.epochs):
                model.train()
                training_loss = 0
                for sequences, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{CFG.epochs} Training'):
                    # print(f"Batch shape: {sequences.shape}")  # debugging line
                    sequences, labels = sequences.to(device), labels.to(device)
                    optimizer.zero_grad()
                    predictions = model(sequences)
                    loss = loss_function(predictions, labels)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()

                # Update learning rate
                lr_scheduler.step()

                # test loop
                model.eval()
                total_test_loss = 0
                with torch.no_grad():
                    for sequences, labels in test_loader:
                        sequences, labels = sequences.to(device), labels.to(device)
                        predictions = model(sequences)
                        test_loss = loss_function(predictions, labels)
                        total_test_loss += test_loss.item()

                # early stopping and logging
                avg_test_loss = total_test_loss / len(test_loader)
                early_stopping(avg_test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

                # log training and validation results
                if CFG.logging:
                    wandb.log({
                        "epoch": epoch,
                        "training loss": training_loss / len(train_loader),
                        "validation loss": avg_test_loss
                    })

        # Save model and log final metrics for fold
        fold += 1
        torch.save(model.state_dict(), f'model_{CFG.version}.pth')
        if CFG.logging:
            wandb.save(f'{CFG.model_name}.pth')

        all_folds_test_losses.append(avg_test_loss)
        all_folds_train_losses.append(training_loss / len(train_loader))

        # calculate and log overall performance
        model_train_loss = sum(all_folds_train_losses) / len(all_folds_train_losses)
        model_test_loss = sum(all_folds_test_losses) / len(all_folds_test_losses)
        if CFG.logging:
            wandb.log({
                "model training loss": model_train_loss,
                "model validation loss": model_test_loss
            })
            wandb.finish()

        print(f"Model train loss: {model_train_loss:.4f}, Model validation loss: {model_test_loss:.4f}")


def run_inference(CFG, model_path, data_loader, scaler):
    """
    run inference on the validation dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scaler = joblib.load(os.path.join(CFG.model_path, 'scaler.pkl'))

    # Load the trained model
    model = LSTMModel(
        input_size=CFG.input_size,
        hidden_layer_size=CFG.hidden_units,
        seq_length=CFG.output_size,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            output = model(sequences)
            predictions.extend(output.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # check if predictions and actuals are not empty
    if predictions:
        # Reshape predictions and actuals for inverse transform
        predictions = np.array(predictions).flatten()  # Flatten to match the actuals' shape
        actuals = np.array(actuals).flatten()
        # inverse transform if scaler is provided

        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        mae = np.mean(np.abs(predictions - actuals))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

        # plot results
        plt.figure(figsize=(10, 5))
        plt.plot(predictions, label='Predicted Demand')
        plt.plot(actuals, label='Actual Demand')
        plt.title('Comparison of Predicted and Actual Electricity Demand')
        plt.xlabel('Time')
        plt.ylabel('Electricity Demand')
        plt.legend()
        plt.savefig(os.path.join(CFG.images_path, 'demand_forecast_vs_actual.png'))
        plt.show()

    else:
        print("No predictions were made.")


if __name__ == "__main__":
    set_seed(LstmCFG.seed)  # for reproducibility
    CFG = LstmCFG()  # load configuration
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data and preprocess
    path = os.path.join(LstmCFG.data_path, 'nsw_df.parquet')
    train_dataset, test_dataset, val_dataset, scaler = preprocess_data(path)
    label_col = 'TOTAL_DEMAND'  # define the label column

    # train model
    if LstmCFG.train:
        train_model(train_dataset, test_dataset, input_features, label_col, LstmCFG())

    # run inference
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size,
            shuffle=False
        )

        # Load the scaler used during training for inverse transformation
        scaler = joblib.load(os.path.join(CFG.model_path, 'scaler.pkl'))

        run_inference(
            CFG,
            os.path.join(CFG.model_path, f'lstm_trained_model_v{CFG.version}.pth'),
            val_loader,
            scaler
        )

# todo:
#  save scalars & basic model
#  make predictions from new model, rebuild model architecture,
#  optimise
#  wandb sweeps to optimise hyperparameters
