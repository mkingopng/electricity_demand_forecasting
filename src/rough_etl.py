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
import time
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
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/ACTUAL_HH/': 'data/operational_demand_actual_hh',  # actual current 30 min
    'http://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_HH/': 'data/operational_demand_actual_hh',  # actual archive 30 min
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/ACTUAL_5MIN/': 'data/public_actual_operational_demand_five_min',  # actual current 5 min

    # 'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/ACTUAL_DAILY/': 'data/operational_demand_actual_daily',  # actual current daily
    # 'http://nemweb.com.au/Reports/Archive/Operational_Demand/ACTUAL_DAILY/': 'data/operational_demand_actual_daily', # actual archive daily
    'http://nemweb.com.au/Reports/CURRENT/Operational_Demand/FORECAST_HH/': 'data/operational_demand_forecast_hh',  # forecast current 30 min
    'http://nemweb.com.au/Reports/Archive/Operational_Demand/FORECAST_HH/': 'data/operational_demand_forecast_hh',  # forecast archive 30 min
    # ROOFTOP PV, ACTUAL AND FORECAST, CURRENT AND ARCHIVE
    # 'https://nemweb.com.au/Reports/Current/ROOFTOP_PV/ACTUAL/': 'data/rooftop_pv_actual',  # rooftop pv actual current
    # 'https://nemweb.com.au/Reports/Archive/ROOFTOP_PV/ACTUAL/': 'data/rooftop_pv_actual',  # rooftop pv actual archive
    # 'https://nemweb.com.au/Reports/Current/ROOFTOP_PV/FORECAST/': 'data/rooftop_pv_forecast',  # rooftop pv forecast current
    # 'https://nemweb.com.au/Reports/Archive/ROOFTOP_PV/FORECAST/': 'data/rooftop_pv_forecast',  # rooftop pv forecast archive
    # DISPATCH SCADA
    # 'https://nemweb.com.au/Reports/Current/Dispatch_SCADA/': 'data/dispatch_SCADA',  # dispatch SCADA current
    # 'https://nemweb.com.au/Reports/Archive/Dispatch_SCADA/': 'data/dispatch_SCADA',  # dispatch SCADA archive
    # PUBLIC PRICES
    'https://nemweb.com.au/Reports/Current/Public_Prices/': 'data/public_prices',  # public prices current
    'https://nemweb.com.au/Reports/Archive/Public_Prices/': 'data/public_prices',  # public prices archive
    # VWAFCAS prices
    # 'https://nemweb.com.au/Reports/Current/Vwa_Fcas_Prices/': 'data/vwa_fcas_prices',  # vwa fcas prices current
    # DISPATCH PRICES PRE AP
    # 'https://nemweb.com.au/Reports/Current/Dispatchprices_PRE_AP/': 'data/dispatch_prices_pre_ap',  # dispatch prices current
    # 'https://nemweb.com.au/Reports/Archive/Dispatchprices_PRE_AP/': 'data/dispatch_prices_pre_ap',  # dispatch prices archive
    # ADJUSTED PRICE REPORTS
    # 'https://nemweb.com.au/Reports/Current/Adjusted_Prices_Reports/': 'data/adjusted_prices_reports',  # adjusted prices reports current
    # 'https://nemweb.com.au/Reports/Archive/Adjusted_Prices_Reports/': 'data/adjusted_prices_reports',  # adjusted prices reports archive
    # TRADING CUMULATIVE PRICE
    # 'https://nemweb.com.au/Reports/Current/Trading_Cumulative_Price/': 'data/trading_cumulative_price',  # trading cumulative price current
    # 'https://nemweb.com.au/Reports/Archive/Trading_Cumulative_Price/': 'data/trading_cumulative_price',  # trading cumulative price archive
    # MKTSUSP PRICING
    # 'https://nemweb.com.au/Reports/Current/Mktsusp_Pricing/': 'data/mktsusp_pricing',  # mktsusp pricing current
}
# trading, settlements, prices, pre-dispatch, PASA, other, network, NEMDE,
# gas supply hub, dispatch, demand and forecasts, bids


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
    Extract zip files that have been downloaded
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
    Organize CSV files into subdirectories based on the file name
    :param source_dir: directory containing the CSV files
    :return: None
    """
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            match = re.match(r'(.*?)(?:_\d{8})', filename)
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
    Remove all empty subdirectories in the given directory
    :param dir_path: The root directory path to search for empty subdirectories
    :return: None
    """
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
                logging.info(f'Removed empty directory: {os.path.join(root, name)}')
            except OSError as e:
                logging.error(f'Error removing directory {os.path.join(root, name)}: {e}')


def consolidate_csv_in_subdirs(data_directory):
    """
    Process and consolidate CSV files in each subdirectory of the data directory.
    """
    for subdir, _, _ in os.walk(data_directory):
        if subdir == data_directory:
            continue
        csv_pattern = os.path.join(subdir, "*.CSV")
        csv_files = glob.glob(csv_pattern)
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
            combined_df.to_csv(consolidated_csv_path, index=False)
            logging.info(f"Saved consolidated CSV for {os.path.basename(subdir)} to {consolidated_csv_path}")
        else:
            logging.info(f"No CSV files found in {subdir}")
        gc.collect()


def write_df_to_postgres(df, table_name, connection_string):
    """
    Write a DataFrame to a PostgreSQL database.
    :param df:
    :param table_name:
    :param db_name:
    :param db_user:
    :param db_pass:
    :param db_host:
    :param db_port:
    :return:
    """
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"DataFrame written to PostgreSQL table {table_name}")


def write_consolidated_csv_to_db(data_directory, connection_string):
    """
    Write the consolidated CSV files to a PostgreSQL database.
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
    # todo: need to clarify data URLs based on what we're using
    extract_zip_files(url_dict_map)
    organize_csv_files(data_directory)
    remove_empty_dirs(data_directory)
    consolidate_csv_in_subdirs(data_directory)
    # todo: need to improve cleaning, file handling & IO
