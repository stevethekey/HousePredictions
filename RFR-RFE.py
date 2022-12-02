"""
Random Forest Regressor with Recursive Feature Elimination
implemented using built-ins
"""
import copy
import time

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # Open the cleaned data
    data = pd.read_csv('cleaned.csv')

    # Create train and test files
    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    # RFE cross validation
    rfecv = RFECV(estimator=RandomForestRegressor(max_depth=6), min_features_to_select=1, n_jobs=-1, scoring="r2")
    rfecv.fit(X_train, y_train)
    y_predicted = rfecv.predict(X_test)
    features_dropped = len(features.columns) - len(features.columns[rfecv.support_])
    # print(features.columns[rfecv.support_])

    # We want inverse of this array
    flipped_support = copy.deepcopy(rfecv.support_)
    for i in range(0, len(flipped_support), 1):
        flipped_support[i] = not flipped_support[i]

    # Perform RFR with the now optimal set, ONLY for timer purposes
    optimal_features = copy.deepcopy(features)
    for dropFeature in features.columns[flipped_support]:
        optimal_features.drop(dropFeature, axis=1)
    o_X_train, o_X_test, o_y_train, o_y_test = train_test_split(optimal_features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)
    rfr = RandomForestRegressor(max_depth=6)
    start = time.perf_counter()
    rfr.fit(o_X_train, o_y_train)
    stop = time.perf_counter()

    # Normalized error
    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error/(max(y_test) - min(y_test))
    # print('Normalized RMSE with {} features dropped: {}'.format(features_dropped, normalized_error))

    # Plotting
    rfrRFE_plot = plt.figure()
    plt.scatter(y_test, y_predicted, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Random Forest Regression with Recursive Feature Elimination\nOptimal Features: {}\nTime to fit the model (in seconds): {:0.4}\nNormalized RMSE: {}'.format(len(features.columns)-features_dropped, stop-start, normalized_error))
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    rfrRFE_plot.set_size_inches(10, 10)
    rfrRFE_plot.savefig('Graphs/RFR_RFE.png', dpi=600)
