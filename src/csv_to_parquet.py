"""
This doesn't work yet!!!
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
    data_path = './../data'
    file_name = 'NSW_consolidated.csv'
    date_fields = ['DATETIME']
    img_dim1 = 20
    img_dim2 = 10


if __name__ == "__main__":
    # Instantiate the DataProcessor with the base data path
    processor = DataProcessor(CFG.data_path)

    # Load the dataset
    totaldemand_nsw = processor.load_csv(
        os.path.join(CFG.data_path, 'NSW', CFG.file_name),
        parse_dates=CFG.date_fields
    )

    # Inspect the loaded data
    print(totaldemand_nsw.head())
    print(totaldemand_nsw.dtypes)

    # Save the dataset in Parquet format
    processor.save_parquet(relative_path=os.path.join(
        CFG.data_path,
        'NSW_consolidated.parquet'))
