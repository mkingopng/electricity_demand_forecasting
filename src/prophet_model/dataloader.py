"""

"""
import pandas as pd
import os


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        # load and preprocess the data
        df = pd.read_parquet(os.path.join(self.config.data_path, 'nsw_df.parquet'))
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ds', 'TOTALDEMAND': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        return df

    def split_data(self, df):
        # split data into train, test, and validation sets
        cutoff_date = df['ds'].max() - pd.Timedelta(days=7)
        cutoff_date2 = df['ds'].max() - pd.Timedelta(days=14)
        df_train = df[df['ds'] <= cutoff_date2]
        df_test = df[(df['ds'] > cutoff_date2) & (df['ds'] <= cutoff_date)]
        df_val = df[df['ds'] > cutoff_date]
        return df_train, df_test, df_val

