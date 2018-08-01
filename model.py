import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import split
import matplotlib.pyplot as plt


params = [1, 3, 5, 6, 7, 8, 9, 10]
test_scores = []
X_train, X_test, y_train, test_ids = split.split()
for param in params:
	clf = RandomForestRegressor(n_estimators=30, max_depth=param)
	test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
	test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title("Param vs CV Error")
plt.show()

