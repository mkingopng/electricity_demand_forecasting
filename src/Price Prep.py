import os
import pandas as pd
import matplotlib.pyplot as plt

data_path = './../Noels_folder/'


def get_file_list(data_path):
    print('******************** : Scan Folder for Files : *******************')
    print(f"Looking in: {data_path}")  # Directly use data_path
    files_to_load = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")
                files_to_load.append(file_path)
    if not files_to_load:
        print("No CSV files found.")
    return files_to_load


def combine_files_to_df(files_to_load):
    """

    :param files_to_load:
    :return:
    """
    print(' ******************** : Load & Combine Data: ********************')
    dfs = []
    loaded = 0
    for file in files_to_load:
        try:
            df = pd.read_csv(file)
            loaded += 1
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if loaded > 0:
        combined_df = pd.concat(dfs)
    else:
        print("No files were loaded.")
        return None
    print(f'Total Files to Load: {len(files_to_load)}')
    print(f'Total Files Loaded: {loaded}')
    if loaded > 0:
        print(f'Total Rows: {len(combined_df)}')
        print(combined_df.shape)
        print(combined_df.columns)
        print(combined_df.dtypes)
    return combined_df


def data_preparation(combined_df):
    """

    :param combined_df:
    :return:
    """
    print(' ******************** : Data Preparation: ********************')
    # df with relevant columns
    combined_df = combined_df[['SETTLEMENTDATE', 'TOTALDEMAND', 'RRP']]
    price_df = combined_df.rename(columns={
        'SETTLEMENTDATE': 'date',
        'TOTALDEMAND': 'totaldemand',
        'RRP': 'rrp'
    })
    print(price_df.columns)

    # convert date
    price_df['date'] = pd.to_datetime(price_df['date'])
    print(price_df['date'].dtypes)

    # aggregate Data to 30 minute intervals
    price_df.set_index('date', inplace=True)
    price_df_clean = price_df.resample('30min').mean().reset_index()
    # price_df_clean.reset_index()

    print(price_df_clean.columns)
    print(len(price_df_clean))
    print(price_df_clean)

    # outliers removed
    price_upper = 2000
    price_lower = -650

    # price outliers
    price_outliers_upper = price_df_clean[(price_df_clean['rrp'] > price_upper)]
    price_outliers_lower = price_df_clean[(price_df_clean['rrp'] < price_lower)]
    price_outliers = pd.concat([price_outliers_upper, price_outliers_lower], axis=0)
    print(f'Total Rows: {len(price_outliers)}')
    # print(price_outliers.describe)

    price_outlier_removed_df = price_df_clean[(price_df_clean['rrp'] < price_upper) & (price_df_clean['rrp'] > price_lower)]
    print(f'Total Rows: {len(price_outlier_removed_df)}')
    # print(price_outlier_removed_df.describe)

    return price_df_clean, price_outliers, price_outlier_removed_df


def data_quality_checks(price_df_clean, price_outliers, price_outlier_removed_df):
    """

    :param price_df_clean:
    :param price_outliers:
    :param price_outlier_removed_df:
    :return:
    """
    print(' ******************** : Data Quality: ********************')
    print(f'Price Data (Clean): {price_df_clean.shape}')
    print(f'Price Data (Outliers Only): {price_outliers.shape}')
    print(f'Price Data (Outliers Removed): {price_outlier_removed_df.shape}')
    # print(price_outlier_removed_df.describe)
    print(price_outliers.shape)
    # print(price_outlier_removed_df.info)
    print(price_outliers.isna().sum())
    print(price_outliers.nunique())


def data_visualisation(price_df_clean, price_outliers, price_outlier_removed_df):
    """

    :param price_df_clean:
    :param price_outliers:
    :param price_outlier_removed_df:
    :return:
    """
    print(' ******************** : Data Visualisation: ********************')
    plt.scatter(price_df_clean['date'],price_df_clean['rrp'])
    plt.show()


def export_data(df, file_name):
    """
    Exports the given DataFrame to CSV and Parquet formats.

    :param df: DataFrame to export.
    :param file_name: Base name for the output file, without extension.
    """
    if df is not None:
        # Construct file paths
        csv_file = os.path.join(data_path, f"{file_name}.csv")
        parquet_file = os.path.join(data_path, f"{file_name}.parquet")

        # Export to CSV
        df.to_csv(csv_file, index=False)
        print(f"Data exported to CSV file: {csv_file}")

        # Export to Parquet
        df.to_parquet(parquet_file, index=False)
        print(f"Data exported to Parquet file: {parquet_file}")
    else:
        print("No data to export.")


if __name__ == '__main__':
    files_to_load = get_file_list(data_path)
    combined_df = combine_files_to_df(files_to_load)
    if combined_df is not None:
        price_df_clean, price_outliers, price_outlier_removed_df = data_preparation(combined_df)
        data_quality_checks(price_df_clean, price_outlier_removed_df, price_outliers)
        data_visualisation(price_df_clean, price_outliers, price_outlier_removed_df)
        export_data(price_df_clean, "./../data/price_cleaned_data")
