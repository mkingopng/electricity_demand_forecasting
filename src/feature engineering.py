import os
import pandas as pd
from scipy.stats import zscore

# pandass display options
pd.options.display.max_columns = 50
pd.options.display.max_rows = 50
pd.options.display.width = 120
pd.options.display.float_format = '{:.2f}'.format

# todo: this needs to be updated to reflect the latest changes in NS_Test.ipynb
class NSWDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.DataFrame()

    def load_csv(self, file_name, parse_dates=None, dayfirst=False, index_col=None, usecols=None):
        """
        loads one or more csv files and returns df
        :param file_name:
        :param parse_dates:
        :param dayfirst:
        :param index_col:
        :param usecols:
        :return:
        """
        full_path = os.path.join(self.data_path, file_name)
        df = pd.read_csv(
            full_path,
            parse_dates=parse_dates,
            dayfirst=dayfirst,
            index_col=index_col,
            usecols=usecols
        )
        # keep only numeric columns for aggregation
        numeric_df = df.select_dtypes(include=['number'])
        # check for duplicate indices and handle if necessary
        if not numeric_df.index.is_unique:
            print(f"Duplicate indices found in {file_name}. Aggregating...")
            numeric_df = numeric_df.groupby(numeric_df.index).mean()
        return numeric_df

    def merge_dfs(self, dfs):
        """
        merges a list of dataframes on their indices.
        :param dfs:
        :return:
        """
        self.df = pd.concat(dfs, axis=1).dropna()

    def smooth_and_backfill(self, column_name, rolling_window=48):
        """
        smooths a column by taking the rolling mean and backfilling the first
        'rolling_window' NaN values.
        :param column_name:
        :param rolling_window:
        :return:
        """
        smoothed_column = self.df[column_name].rolling(window=rolling_window).mean()
        # backfill the first 'rolling_window' NaN values
        smoothed_column[:rolling_window] = smoothed_column[rolling_window:].bfill().head(rolling_window)
        self.df[f'smoothed_{column_name}'] = smoothed_column

    def resample_and_fill(self, column_name, freq='D'):
        """
        resamples a column to a specified frequency and fills missing values
        :param column_name:
        :param freq:
        :return:
        """
        resampled_series = self.df[column_name].resample(freq).mean()
        reindexed_series = resampled_series.reindex(self.df.index, method='bfill')
        self.df[f'{column_name}_daily'] = reindexed_series

    def normalize_columns(self, columns):
        """
        applies z-score normalization to specified columns
        :param columns:
        :return:
        """
        for column in columns:
            self.df[f'normalized_{column}'] = zscore(self.df[column])

    def process(self):
        """
        executes the data processing pipeline
        :return:
        """
        forecast_demand = self.load_csv('forecastdemand_nsw.csv', parse_dates=['LASTCHANGED', 'DATETIME'], index_col='DATETIME')
        total_demand = self.load_csv('totaldemand_nsw.csv', parse_dates=['DATETIME'], dayfirst=True, index_col='DATETIME')
        temperature = self.load_csv('temperature_nsw.csv', parse_dates=['DATETIME'], dayfirst=True, index_col='DATETIME')
        price_df = self.load_csv('price_cleaned_data.csv', parse_dates=['date'], index_col='date')

        # merge dfs
        self.merge_dfs([forecast_demand, total_demand, temperature, price_df])

        # smooth and backfill
        self.smooth_and_backfill('FORECASTDEMAND')
        self.smooth_and_backfill('TOTALDEMAND')
        self.smooth_and_backfill('TEMPERATURE')

        # resample and fill
        self.resample_and_fill('FORECASTDEMAND')
        self.resample_and_fill('TOTALDEMAND')

        # normalize
        self.normalize_columns(['TOTALDEMAND', 'FORECASTDEMAND', 'TEMPERATURE'])
        return self.df


if __name__ == "__main__":
    data_processor = NSWDataProcessor('./../data/NSW')
    processed_df = data_processor.process()  # store returned df as var

    # select a subset of columns
    subset_df = processed_df[
        ['FORECASTDEMAND', 'TOTALDEMAND', 'TEMPERATURE', 'rrp',
         'smoothed_FORECASTDEMAND', 'smoothed_TOTALDEMAND',
         'smoothed_TEMPERATURE', 'FORECASTDEMAND_daily', 'TOTALDEMAND_daily',
         'normalized_TOTALDEMAND', 'normalized_FORECASTDEMAND',
         'normalized_TEMPERATURE']
    ]

    print(subset_df.head())  # print head of subset df

    # save subset df to CSV and Parquet
    subset_df.to_csv('./../data/NSW/processed_data_subset.csv')
    subset_df.to_parquet('./../data/NSW/processed_data_subset.parquet')
