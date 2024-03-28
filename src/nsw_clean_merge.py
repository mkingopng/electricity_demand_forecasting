"""

"""
import pandas as pd
import os
import matplotlib.pyplot as plt


class CFG:
    data_path = './../data'
    img_dim1 = 20
    img_dim2 = 10


nsw_forecastdemand = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'forecastdemand_nsw.csv')
)

nsw_temperature = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'temperature_nsw.csv')
)

nsw_totaldemand = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'totaldemand_nsw.csv')
)

# step 1 merge
merged_df = pd.merge(
    nsw_totaldemand[['DATETIME', 'TOTALDEMAND', 'REGIONID']],
    nsw_forecastdemand[['PREDISPATCHSEQNO', 'REGIONID', 'PERIODID', 'FORECASTDEMAND', 'LASTCHANGED', 'DATETIME']],
    on=['DATETIME', 'REGIONID'],
    how='outer'
)

# step 2 merge
nsw_df = pd.merge(
    merged_df,
    nsw_temperature[['LOCATION', 'DATETIME', 'TEMPERATURE']],
    on='DATETIME',
    how='inner'
)

print(nsw_df.head())

nsw_df.to_csv(os.path.join(CFG.data_path, 'NSW', 'nsw_data.csv'))
nsw_df.to_parquet(os.path.join(CFG.data_path, 'NSW', 'nsw_data.parquet'))
