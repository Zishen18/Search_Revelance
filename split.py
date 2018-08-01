import numpy as np
import pandas as pd

def split():

	df_train = pd.read_csv('../Home_Depot_data/train.csv', encoding="ISO-8859-1")
	df_test = pd.read_csv('../Home_Depot_data/test.csv', encoding="ISO-8859-1")
	df_all = pd.read_csv('df_all_feature.csv', encoding="ISO-8859-1")

	df_train = df_all.loc[df_train.index]
	df_test = df_all.loc[df_test.index]

	test_ids = df_test['id']
	y_train = df_train['relevance'].values

	X_train = df_train.drop(['id', 'relevance'], axis=1).values
	X_test = df_test.drop(['id', 'relevance'], axis=1).values

	return X_train, X_test, y_train, test_ids	
