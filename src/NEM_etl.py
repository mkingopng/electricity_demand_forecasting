"""
This script
- downloads the zip files from a list of URLs and
- extracts zip file to csv and clean up
- organizes the CSV files into subdirectories
- cleans and consolidates into smallest number of dataframes
"""
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import logging
from tqdm import tqdm
import zipfile
from zipfile import BadZipFile
from datetime import date
import shutil
import re
import pandas as pd
import gc
import glob
from sqlalchemy import create_engine
from dotenv import load_dotenv

today = date.today()
data_directory = './../data'
log_filename = f'./../logs/log_{today}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

url_dict_map = {
    # OPERATIONAL DEMAND
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/ACTUAL_HH/':
        f'{data_directory}/operational_demand_actual_hh',  # current 30 min
    'http://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/':
        f'{data_directory}/operational_demand_actual_hh',  # archive 30 min
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/ACTUAL_5MIN/':
        f'{data_directory}/public_actual_operational_demand_five_min',  # actual current 5 min

    # FORECAST DEMAND
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/FORECAST_HH/':
        f'{data_directory}/operational_demand_forecast_hh',  # forecast 30 min
    'http://nemweb.com.au/Reports/Archive/Operational_Demand/FORECAST_HH/':
        f'{data_directory}/operational_demand_forecast_hh',  # forecast 30 min

    # PUBLIC PRICES
    'https://nemweb.com.au/Reports/Current/Public_Prices/':
        f'{data_directory}/public_prices',  # public prices current
    'https://nemweb.com.au/Reports/Archive/Public_Prices/':
        f'{data_directory}/public_prices',  # public prices archive

###############

    # ROOFTOP PV, ACTUAL AND FORECAST, CURRENT AND ARCHIVE
    # 'https://nemweb.com.au/Reports/Current/ROOFTOP_PV/ACTUAL/':
    #     f'{data_directory}/rooftop_pv_actual',
    # 'https://nemweb.com.au/Reports/Archive/ROOFTOP_PV/ACTUAL/':
    #     f'{data_directory}/rooftop_pv_actual',
    # 'https://nemweb.com.au/Reports/Current/ROOFTOP_PV/FORECAST/':
    #     f'{data_directory}/rooftop_pv_forecast',
    # 'https://nemweb.com.au/Reports/Archive/ROOFTOP_PV/FORECAST/':
    #     f'{data_directory}/rooftop_pv_forecast',

    # Hist Demand
    # 'https://nemweb.com.au/Reports/Current/HistDemand/':
    #     f'{data_directory}/hist_demand',
    # 'https://nemweb.com.au/Reports/ARCHIVE/HistDemand/':
    #     f'{data_directory}/hist_demand',

    # seven day outlook full
    # 'https://nemweb.com.au/Reports/Current/SEVENDAYOUTLOOK_FULL/':
    #     f'{data_directory}/seven_day_outlook_full',
    # 'https://nemweb.com.au/Reports/ARCHIVE/SEVENDAYOUTLOOK_FULL/':
    #     f'{data_directory}/seven_day_outlook_full',

    # DISPATCH scada
    # 'https://nemweb.com.au/Reports/Current/Dispatch_SCADA/':
    #     f'{data_directory}/dispatch_scada',
    # 'https://nemweb.com.au/Reports/ARCHIVE/Dispatch_SCADA/':
    #     f'{data_directory}/dispatch_scada',
}


def download_zip_files(url_directory_map):
    """
    Download zip files from a dictionary of URLs and their corresponding
    download folders
    :param url_directory_map:
    :return: None
    """
    for url, download_folder in url_directory_map.items():
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        zip_links = [link for link in soup.find_all('a') if link.get('href')
                     and link.get('href').endswith('.zip')]
        for link in tqdm(zip_links, desc=f"Downloading from {url}"):
            href = link.get('href')
            download_url = urljoin(url, href)
            filename = os.path.join(download_folder, os.path.basename(href))
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                # todo: log exception
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


def extract_zip_files(url_directory_map):
    """
    extract zip files that have been downloaded
    :param url_directory_map:
    :return: None
    """
    for _, download_folder in url_directory_map.items():
        zip_files = [item for item in os.listdir(download_folder)
                     if item.endswith('.zip')]
        for item in tqdm(zip_files, desc=f"Xtracting to: {download_folder}"):
            if item.endswith('.zip'):
                filename = os.path.join(download_folder, item)
                try:
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(download_folder)
                        logging.info(f'Extracted {filename}')
                    os.remove(filename)
                except BadZipFile:
                    logging.error(f'Bad zip file: {filename}')
                except Exception as e:
                    logging.error(f'Error extracting {filename}: {e}')


def organize_csv_files(source_dir):
    """
    organize CSV files into subdirectories based on the file name
    :param source_dir: directory containing the CSV files
    :return: None
    """
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            match = re.match(r'(.*?)_\d{8}', filename)
            if match:
                dir_name = match.group(1).lower().replace(" ", "_")
                target_dir_path = os.path.join(source_dir, dir_name)
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)

                shutil.move(
                    os.path.join(source_dir, filename),
                    os.path.join(target_dir_path, filename)
                )


def remove_empty_dirs(dir_path):
    """
    remove empty directories
    :param dir_path:
    """
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):  # Checks if the directory is empty
                try:
                    os.rmdir(dir_path)
                    logging.info(f"Removed empty directory: {dir_path}")
                except OSError as e:
                    logging.error(f"Error removing directory {dir_path}: {e}")
            else:
                logging.info(f"Directory not empty: {dir_path}")


def consolidate_csv_in_subdirs(data_directory):
    """
    process and consolidate CSV files in each subdir of the data directory.
    :param data_directory:
    :return:
    """
    for subdir, _, _ in os.walk(data_directory):
        if subdir == data_directory:
            continue
        csv_pattern = os.path.join(subdir, "*.[cC][sS][vV]")
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            logging.info(f"No CSV files found in {subdir}, skipping...")
            continue
        dfs = []
        for filename in tqdm(csv_files, desc=f"Processing {os.path.basename(subdir)}"):
            try:
                df = pd.read_csv(filename, header=1, skipfooter=1, engine='python', on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            consolidated_csv_path = os.path.join(data_directory, f"{os.path.basename(subdir)}_consolidated.csv")
            combined_df.to_parquet(
                consolidated_csv_path,
                index=False,
                engine='pyarrow',
                compression='snappy'
            )
            logging.info(f"Saved consolidated DF for {os.path.basename(subdir)} to {consolidated_csv_path}")
            for filename in csv_files:
                os.remove(filename)
                logging.info(f"Deleted {filename}")
        else:
            logging.info(f"No CSV files found in {subdir}")
        gc.collect()


def write_df_to_postgres(df, table_name, connection_string):
    """
    write df to PostgreSQL db.
    :param connection_string:
    :param df:
    :param table_name:
    :return:
    """
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"DataFrame written to PostgreSQL table {table_name}")


def write_consolidated_csv_to_db(data_directory, connection_string):
    """
    write the consolidated CSV files to a PostgreSQL database.
    :param data_directory:
    :param connection_string:
    :return:
    """
    consolidated_files_pattern = os.path.join(data_directory, "*_consolidated.csv")
    consolidated_files = glob.glob(consolidated_files_pattern)

    for filepath in tqdm(consolidated_files, desc="writing CSVs to PostgreSQL"):
        table_name = os.path.basename(filepath).replace('_consolidated.csv', '')
        df = pd.read_csv(filepath)
        write_df_to_postgres(df, table_name, connection_string)


if __name__ == "__main__":
    load_dotenv()
    download_zip_files(url_dict_map)
    extract_zip_files(url_dict_map)
    organize_csv_files(data_directory)
    remove_empty_dirs(data_directory)
    consolidate_csv_in_subdirs(data_directory)
    remove_empty_dirs(data_directory)

# todo: exclude project files,
#  add additional information for variables,
