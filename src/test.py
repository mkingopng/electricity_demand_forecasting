import pandas as pd
import os


class CFG:
    data_path = './../data'
    img_dim1 = 20
    img_dim2 = 10


# df = pd.read_csv('./../data/NSW/nsw_data.csv')
df = pd.read_csv(os.path.join(CFG.data_path, 'NSW', 'nsw_data.csv'))

print(df.head())
