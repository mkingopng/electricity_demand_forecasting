import pandas as pd
import numpy as np
from plot_settings import *
from sys import path

nsw_df = pd.read_parquet("../data/NSW/nsw_df.parquet")

new = pd.read_csv("dataRB/totaldemand_part2.csv")
new['DATETIME'] = pd.to_datetime(new['DATETIME'])
new.info()

nsw_df_to2024 = pd.concat([nsw_df, new])
nsw_df_to2024.to_parquet("../data/NSW/nsw_df_to2024")

##### New temp data
tmp = pd.read_csv("dataRB/historical_temp_data.csv")
tmp