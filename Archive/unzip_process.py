import os
import zipfile

# create the 'data' directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Path to the directory where files are stored
directory_path = '../data'


def csv_files_exist(directory):
    """
    function to check if CSV files already exist
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return len(csv_files) > 0


def unzip_file(zip_path, extract_to):
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
    else:
        print(f"Error: Zip file {zip_path} does not exist.")


# check if CSV files already exist
if not csv_files_exist(directory_path):
    # concatenate parts of the zip file if the CSV files do not exist
    parts = sorted([f for f in os.listdir(directory_path) if
                    f.startswith('forecastdemand_nsw.csv.zip.part')])
    full_zip_path = os.path.join(directory_path, 'forecastdemand_nsw_csv.zip')
    with open(full_zip_path, 'wb') as full_zip:
        for part in parts:
            part_path = os.path.join(directory_path, part)
            with open(part_path, 'rb') as file_part:
                full_zip.write(file_part.read())

    # unzip the files
    unzip_file(full_zip_path, directory_path)
    unzip_file('../data/NSW/temperature_nsw_csv.zip', directory_path)
    unzip_file('../data/NSW/totaldemand_nsw_csv.zip', directory_path)
    unzip_file('../data/NSW/forecastdemand_nsw_csv.zip', directory_path)
else:
    print("CSV files already exist. Skipping unzip operations.")
