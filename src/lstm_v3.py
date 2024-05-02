"""

"""
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class LstmCFG:
    """
    configuration class for the LSTM model
    """
    wandb_project_name = 'electricity_demand_forecasting'
    wandb_run_name = 'lstm'
    data_path = './../data/NSW'
    image_path = './../images'
    logging = True
    train = False
    version = 1
    n_folds = 10
    epochs = 30
    # n_features = 33
    # input_size = 33
    num_layers = 5
    hidden_units = 50
    output_size = 1
    lr = 0.0003
    batch_size = 1024
    seq_length = 336  # 336 one week of 30-minute sample intervals
    dropout = 0.2
    weight_decay = 0.00001
    lrs_step_size = 6
    lrs_gamma = 0.4
    target_feature = 'TOTALDEMAND'
    input_features = [  # 25
        'TOTALDEMAND',  # continuous
        'FORECASTDEMAND',  # continuous
        'TEMPERATURE',  # continuous
        'rrp',  # continuous
        'daily_avg_actual',  # continuous
        'daily_avg_forecast',  # continuous
        'forecast_error',  # continuous
        'smoothed_forecast_demand',  # continuous
        'year',  # cyclical
        'quarter',  # cyclical
        'month',  # cyclical
        'week_of_year',  # cyclical
        'dow',  # cyclical
        'doy',  # cyclical
        'day_of_month',  # cyclical
        'hour',  # cyclical
        'is_weekend',  # boolean, categorical
        'part_of_day',  # categorical
        'season',  # categorical
        'is_business_day',  # boolean, categorical
        'smoothed_total_demand',  # continuous
        'smoothed_temperature',  # continuous
        'minutes_past_midnight',  # cyclical
        'season_name'  # categorical
    ]
    continuous_features = [  # 10
        'TOTALDEMAND',
        'FORECASTDEMAND',
        'TEMPERATURE',
        'rrp',
        'daily_avg_actual',
        'daily_avg_forecast',
        'forecast_error',
        'smoothed_forecast_demand',
        'smoothed_total_demand',
        'smoothed_temperature'
    ]
    cyclical_features = [  # 9
        'hour',
        'dow',
        'doy',
        'month',
        'quarter',
        'week_of_year',
        'minutes_past_midnight'
    ]
    categorical_features = [  # 5
        'is_weekend',
        'part_of_day',
        'season',
        'is_business_day',
        'season_name'
    ]
    max_values = {  # define the maximum values for each cyclical feature
        'year': 2021,
        'quarter': 4,
        'month': 12,
        'week_of_year': 52,
        'dow': 7,
        'doy': 365,  # todo: 366 for leap years to be more precise
        'day_of_month': 31,
        'hour': 24,
        'minutes_past_midnight': 1439
    }
    # wandb_config = {
    #     "n_folds": LstmCFG.n_folds,
    #     "n_features": LstmCFG.input_size,
    #     "hidden layers": LstmCFG.hidden_units,
    #     "learning_rate": LstmCFG.lr,
    #     "batch_size": LstmCFG.batch_size,
    #     "epochs": LstmCFG.epochs,
    #     "sequence_length": LstmCFG.seq_length,
    #     "dropout": LstmCFG.dropout,
    #     "num_layers": LstmCFG.num_layers,
    #     "weight_decay": LstmCFG.weight_decay,
    #     "lrs_step_size": LstmCFG.lrs_step_size,
    #     "lrs_gamma": LstmCFG.lrs_gamma,
    # }


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.column_transformer = ColumnTransformer([
            ("cyclical", FunctionTransformer(self.encode_cyclical_features, validate=False), self.config.cyclical_features),
            ("categorical", OneHotEncoder(handle_unknown='ignore'), self.config.categorical_features),
            ("continuous", MinMaxScaler(), self.config.continuous_features)
        ], remainder='passthrough')

    def encode_cyclical_features(self, x):
        for column in self.config.cyclical_features:
            max_value = self.config.max_values[column]
            x[column + '_sin'] = np.sin(2 * np.pi * x[column] / max_value)
            x[column + '_cos'] = np.cos(2 * np.pi * x[column] / max_value)
        print("After cyclical encoding:", x.head())  # Print sample data after transformation
        return x

    def get_feature_names(self):
        feature_names = []
        # Ensure the ColumnTransformer has been fitted
        if not hasattr(self.column_transformer, 'named_transformers_'):
            raise RuntimeError(
                "The column transformer has not been fitted yet.")

        # Loop through all the transformers applied in the ColumnTransformer
        for name, transformer, _ in self.column_transformer.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    # If transformer supports it, get feature names directly
                    transformed_features = transformer.get_feature_names_out()
                    feature_names.extend(
                        [f"{name}__{feat}" for feat in transformed_features])
                except Exception as e:
                    print(
                        f"Error handling get_feature_names_out for {name}: {str(e)}")
                    feature_names.extend(
                        [f"{name}__{col}" for col in _ if _ is not None])
            else:
                # If no specific feature name method is available, use the column names
                # For FunctionTransformer or similar, directly use the input column names
                if isinstance(transformer, FunctionTransformer):
                    feature_names.extend([f"{name}__{col}_sin" for col in _])
                    feature_names.extend([f"{name}__{col}_cos" for col in _])
                else:
                    feature_names.extend(
                        [f"{name}__{col}" for col in _ if _ is not None])

        return feature_names

    def preprocess_data(self, path, cutoff_days_test=14, cutoff_days_val=28):
        df = pd.read_parquet(path)
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        print("Initial data shape:", df.shape)

        # Define cutoff dates and split data
        cutoff_date_test = df.index.max() - pd.Timedelta(days=cutoff_days_test)
        cutoff_date_val = df.index.max() - pd.Timedelta(days=cutoff_days_val)

        train_df = df[df.index <= cutoff_date_val].copy()
        test_df = df[(df.index > cutoff_date_val) & (
                    df.index <= cutoff_date_test)].copy()
        val_df = df[df.index > cutoff_date_test].copy()

        # Fit the transformer and then retrieve the feature names
        self.column_transformer.fit(train_df)
        feature_names = self.get_feature_names()  # This should be called after fitting
        print("Generated feature names:", feature_names)

        transformed_train = self.column_transformer.transform(train_df)
        print("Transformed train shape:", transformed_train.shape)
        print("Feature names count:", len(feature_names))

        if transformed_train.shape[1] != len(feature_names):
            raise ValueError(
                "Mismatch in the number of feature names and transformed columns")

        train_df = pd.DataFrame(transformed_train, columns=feature_names,
                                index=train_df.index)
        test_df = pd.DataFrame(self.column_transformer.transform(test_df),
                               columns=feature_names, index=test_df.index)
        val_df = pd.DataFrame(self.column_transformer.transform(val_df),
                              columns=feature_names, index=val_df.index)

        return train_df, test_df, val_df


class DemandDataset(Dataset):
    def __init__(
            self, df, label_col, sequence_length=LstmCFG.seq_length,
            forecast_horizon=336
            ):
        """
        Initializes the dataset with the dataframe, label column, sequence length, and forecast horizon.
        Converts dataframe to tensors for efficient data loading.
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Convert the DataFrame to tensors for efficient slicing in __getitem__
        self.features = torch.tensor(df.drop(label_col, axis=1).values,
                                     dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values,
                                   dtype=torch.float32).view(-1, 1)

    def __len__(self):
        """Returns the total number of samples that can be generated from the dataset."""
        total_length = len(
            self.features) - self.sequence_length - self.forecast_horizon + 1
        return max(0, total_length)

    def __getitem__(self, index):
        """Generates one sample of data."""
        sequence_end = index + self.sequence_length
        label_end = sequence_end + self.forecast_horizon

        if sequence_end >= len(self.features) or label_end > len(self.labels):
            raise IndexError("Index range out of bounds for sequence or labels generation.")

        sequence = self.features[index:sequence_end]
        labels = self.labels[sequence_end:label_end]

        return sequence, labels


if __name__ == "__main__":
    config = LstmCFG()  # assuming LstmCFG is already defined and loaded
    preprocessor = DataPreprocessor(config)

    # Load and preprocess the data
    try:
        train_data, test_data, val_data = preprocessor.preprocess_data(
            os.path.join(LstmCFG.data_path, 'nsw_df.parquet'))
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print("An error occurred during preprocessing:", str(e))
        raise  # Re-raise the exception to stop execution if data loading fails

    # Initialize datasets
    train_dataset = DemandDataset(train_data, 'TOTALDEMAND')
    test_dataset = DemandDataset(test_data, 'TOTALDEMAND')
    val_dataset = DemandDataset(val_data, 'TOTALDEMAND')

    # Data loaders for handling data in batches during training
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print("test complete")

    # Here you would typically have the rest of your training pipeline
    # For example, initializing the model, defining loss function, optimizer, etc.

    # Example of accessing a batch of data
    # for sequences, labels in train_loader:
    #     print("Sequence batch shape:", sequences.shape)
    #     print("Label batch shape:", labels.shape)
    #     break  # Just show the first batch and break

    # Continue with your model training and validation
