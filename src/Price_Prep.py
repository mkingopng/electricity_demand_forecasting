import os
import pandas as pd
import matplotlib.pyplot as plt

data_path = './../Noels_folder'

def GetFileList(data_path):
    print(' ******************** : Scan Folder for Files : ********************')
    price_data_folder = os.path.join(data_path,'P_Folder')
    files_to_load = []

    for root, dir, files in os.walk(price_data_folder):
        for file in files:
            if file.endswith('.csv'):
                files_to_load.append(os.path.join(root,file))

    return files_to_load

def CombineFilesToDF(files_to_load):
    print(' ******************** : Load & Combine Data: ********************')
    dfs = []
    loaded =0
    for file in files_to_load:
        df = pd.read_csv(file)
        loaded +=1
        dfs.append(df)
        combined_df = pd.concat(dfs)

    print(f'Total Files to Load: {len(files_to_load)}')
    print(f'Total Files Loaded: {loaded}')
    print(f'Total Rows: {len(combined_df)}')
    print(combined_df.shape)
    print(combined_df.columns)
    print(combined_df.dtypes)
    return combined_df

def DataPreparation(combined_df):
    print(' ******************** : Data Preparation: ********************')
    #DF with relevant columns
    combined_df = combined_df[['SETTLEMENTDATE','TOTALDEMAND','RRP']]
    price_df = combined_df.rename(columns={'SETTLEMENTDATE':'date','TOTALDEMAND':'totaldemand','RRP':'rrp'})
    print(price_df.columns)

    #Convert Date
    price_df['date'] = pd.to_datetime(price_df['date'])
    print(price_df['date'].dtypes)

    #Aggregate Data to 30 minute intervals
    price_df.set_index('date',inplace=True)
    price_df_clean = price_df.resample('30min').mean().reset_index()
    # price_df_clean.reset_index()

    print(price_df_clean.columns)
    print(len(price_df_clean))
    print(price_df_clean)

    #Outliers Removed
    price_upper = 2000
    price_lower = -650

    #Price Outliers
    price_outliers_upper = price_df_clean[(price_df_clean['rrp'] > price_upper)]
    price_outliers_lower = price_df_clean[(price_df_clean['rrp'] < price_lower)]
    price_outliers = pd.concat([price_outliers_upper,price_outliers_lower],axis=0)
    print(f'Total Rows: {len(price_outliers)}')
    # print(price_outliers.describe)

    price_outlier_removed_df = price_df_clean[(price_df_clean['rrp'] < price_upper) & (price_df_clean['rrp']>price_lower)]
    print(f'Total Rows: {len(price_outlier_removed_df)}')
    # print(price_outlier_removed_df.describe)

    return price_df_clean, price_outliers, price_outlier_removed_df

def DataVisualisation(price_df_clean, price_outliers, price_outlier_removed_df):
    print(' ******************** : Data Visualisation: ********************')

    plt.scatter(price_outlier_removed_df['date'],price_outlier_removed_df['rrp'])
    # plt.title('Price Data (Outliers Removed')
    # plt.x_label('Date')
    # plt.y_label('Price')
    plt.show()


if __name__=='__main__':
   files_to_load =  GetFileList(data_path)
   combined_df = CombineFilesToDF(files_to_load)
   price_df_clean, price_outliers, price_outlier_removed_df = DataPreparation(combined_df)
   DataVisualisation(price_df_clean, price_outliers, price_outlier_removed_df)