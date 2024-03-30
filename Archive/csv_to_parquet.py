"""
This works but is clunky and kind sucks
"""
import pandas as pd
import os


class DataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_csv(self, relative_path, parse_dates=None):
        """Load a CSV file into a DataFrame."""
        file_path = os.path.join(
            self.base_path,
            relative_path
        )
        return pd.read_csv(file_path, parse_dates=parse_dates)

    def save_parquet(self, df, relative_path):
        """Save a DataFrame to a Parquet file."""
        file_path = os.path.join(self.base_path, relative_path)
        df.to_parquet(file_path)


class CFG:
    data_path = '../data'
    file_name = '../data/seven_day_outlook_full_consolidated.csv'
    date_fields = ['CALENDAR_DATE', 'INTERVAL_DATETIME']
    img_dim1 = 20
    img_dim2 = 10


if __name__ == "__main__":
    processor = DataProcessor(CFG.data_path)
    df = processor.load_csv(
        os.path.join(CFG.file_name),
        parse_dates=CFG.date_fields
    )

    print(df.head())
    print(df.dtypes)

    processor.save_parquet(
        df,
        os.path.join(f'seven_day_outlook_full_consolidated.parquet')
    )
