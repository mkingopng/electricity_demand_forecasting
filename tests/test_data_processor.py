import pandas as pd
import numpy as np
import os
from src.lstm_v3 import DataPreprocessor, LstmCFG  # Import from module


def test_encode_cyclical_features():
    config = LstmCFG()  # ensure LstmCFG is defined and has the necessary attributes
    preprocessor = DataPreprocessor(config)

    # create a sample DataFrame
    data = pd.DataFrame({
        'hour': [0, 6, 12, 18],
        'dow': [0, 1, 2, 3]
    })

    # Applying cyclical encoding
    transformed = preprocessor.encode_cyclical_features(data.values)
    assert transformed.shape == (4, 8)
    # check the shape is correct after adding sin & cos columns for ea. feature


def test_preprocess_data():
    # ensure LstmCFG is defined and has the necessary attributes
    config = LstmCFG()
    preprocessor = DataPreprocessor(config)
    # mocking the data or using a fixture to load test data
    path = os.path.join(LstmCFG.data_path, 'nsw_df.parquet')  # this should point to a real or mocked file in your testing environment
    print("Checking path:", path)  # This will show the full path being accessed
    # running preprocessing
    try:
        train_df, test_df, val_df = preprocessor.preprocess_data(path)
        assert not train_df.empty and not test_df.empty and not val_df.empty, "Datasets should not be empty"
    except Exception as e:
        assert False, f"Preprocessing failed: {str(e)}"
