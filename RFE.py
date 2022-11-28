import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from math import sqrt

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # open the cleaned data
    data = pd.read_csv('cleaned.csv')

    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    rfr = RandomForestRegressor(random_state=0)
    rfr.fit(X_train, y_train)
    y_predicted = rfr.predict(X_test)

    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error/(max(y_predicted) - min(y_predicted))
    print(normalized_error)
