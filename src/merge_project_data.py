"""

"""
import os
import pandas as pd
import functools


class CFG:
    data_path = './../data'
    img_dim1 = 20
    img_dim2 = 10


nsw_forecastdemand = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'forecastdemand_nsw.csv'),
    parse_dates=['LASTCHANGED', 'DATETIME']
)

nsw_forecastdemand.set_index('DATETIME', inplace=True)

nsw_totaldemand = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'totaldemand_nsw.csv'),
    parse_dates=['DATETIME'],
    dayfirst=True,
    index_col='DATETIME',
    usecols=['DATETIME', 'TOTALDEMAND']
)

nsw_temperature = pd.read_csv(
    os.path.join(CFG.data_path, 'NSW', 'temperature_nsw.csv'),
    parse_dates=['DATETIME'],
    dayfirst=True,
    index_col='DATETIME',
    usecols=['DATETIME', 'TEMPERATURE']
)

# merge
dfs = [nsw_totaldemand, nsw_forecastdemand, nsw_temperature]

nsw_df = functools.reduce(
    lambda left, right:
    pd.merge(
        left,
        right,
        left_index=True,
        right_index=True,
        how='inner'
    ), dfs
)

nsw_df.to_csv(os.path.join(CFG.data_path, 'NSW', 'nsw_data.csv'))
nsw_df.to_parquet(os.path.join(CFG.data_path, 'NSW', 'nsw_data.parquet'))
