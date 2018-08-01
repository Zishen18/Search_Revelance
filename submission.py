import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import split

X_train, X_test, y_train, test_ids = split.split()
rf = RandomForestRegressor(n_estimators=30, max_depth=7)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('submission.csv', index=False)
