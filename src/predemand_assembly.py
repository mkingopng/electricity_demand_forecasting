"""
assemble historic csv files into a single combined df, save to parquet and csv
"""
import pandas as pd
import os
import re
from tqdm import tqdm


class CSVProcessor:
    """
    Process CSV files in a directory, combine them into a single DataFrame,
    and save to Parquet or CSV.
    """
    def __init__(self, data_path, file_pattern=r'.*\.csv$', parse_dates=None):
        self.data_path = data_path
        self.file_pattern = re.compile(file_pattern, re.IGNORECASE)
        self.parse_dates = parse_dates

    def _get_csv_files(self):
        """Retrieve a list of CSV files based on the specified pattern."""
        return [f for f in os.listdir(self.data_path) if self.file_pattern.match(f)]

    def _load_csv(self, filename):
        """Load a single CSV file into a DataFrame."""
        filepath = os.path.join(self.data_path, filename)
        return pd.read_csv(
            filepath,
            header=1,
            skipfooter=1,
            engine='python',
            parse_dates=self.parse_dates
        )

    def load_and_combine_csvs(self):
        """Load and combine CSV files into a single DataFrame."""
        csv_files = self._get_csv_files()
        dfs = [self._load_csv(filename) for filename in tqdm(csv_files, desc="Reading CSV files")]
        return pd.concat(dfs, ignore_index=True)

    def save_to_parquet(self, combined_df, output_path):
        """Save the combined DataFrame to a Parquet file."""
        combined_df.to_parquet(output_path)

    def save_to_csv(self, combined_df, output_path):
        """Save the combined DataFrame to a CSV file."""
        combined_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    data_path = './../data/Public_DVD_PerDemand'

    processor = CSVProcessor(
        data_path=data_path,
        parse_dates=['EFFECTIVEDATE', 'SETTLEMENTDATE', 'OFFERDATE']
    )

    combined_df = processor.load_and_combine_csvs()

    processor.save_to_parquet(
        combined_df,
        os.path.join(data_path, 'Combined_PUBLIC_DVD_PERDEMAND.parquet')
    )

    processor.save_to_csv(
        combined_df,
        os.path.join(data_path, 'Combined_PUBLIC_DVD_PERDEMAND.csv')
    )
