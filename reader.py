import numpy as np
import pandas as pd


df_train = pd.read_csv('../Home_Depot_data/train.csv')
df_test = pd.read_csv('../Home_Depot_data/test.csv')

df_desc = pd.read_csv('../Home_Depot_data/product_descriptions.csv')

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
df_all.to_csv('df_all.csv')

