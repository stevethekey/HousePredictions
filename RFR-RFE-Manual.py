"""
Random Forest Regressor with Recursive Feature Elimination
except the RFE is partial built ins and implemented manually
"""
import copy
import time
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
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

    # Recursive feature elimination
    ranking_ordered = list()
    RFErfr = RandomForestRegressor(max_depth=6)
    rfe = RFE(RFErfr, n_features_to_select=1)
    rfe.fit(X_train, y_train)
    featureCols = features.columns.tolist()
    for x, y in (sorted(zip(rfe.ranking_, featureCols), key=itemgetter(0))):
        ranking_ordered.append(y)
    # print(ranking_ordered)

    y_predicted_optimal = list()
    numFeatures = len(features.columns)
    features_dropped_optimal = 1
    optimal_features = copy.deepcopy(features)
    optimal_rmse = 1
    threshold = 0.001
    features_dropped=0
    base_rmse = 0
    while len(ranking_ordered) > 1:
        rfr = RandomForestRegressor(max_depth=6)
        features = features.drop(ranking_ordered.pop(), axis=1)
        average_normalized_error = 0
        features_dropped += 1
        for _ in range(3):
            X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                                random_state=42)
            rfr.fit(X_train, y_train)
            y_predicted = rfr.predict(X_test)

            # Normalized error
            error = sqrt(mean_squared_error(y_test, y_predicted))
            normalized_error = error/(max(y_test) - min(y_test))
            average_normalized_error += normalized_error
        # print('Average Normalized RMSE with {} features dropped: {}'.format(features_dropped, average_normalized_error/3))
        if features_dropped == 1:
            base_rmse = average_normalized_error/3
        if average_normalized_error/3 < base_rmse+threshold:
            y_predicted_optimal = copy.deepcopy(y_predicted)
            optimal_features = copy.deepcopy(features)
            optimal_rmse = average_normalized_error/3
            features_dropped_optimal = features_dropped
    # print("Optimal features dropped: {}".format(features_dropped_optimal))

    # Calculate timer with optimal features
    o_X_train, o_X_test, o_y_train, o_y_test = train_test_split(optimal_features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)
    o_rfr = RandomForestRegressor(max_depth=6)
    start = time.perf_counter()
    o_rfr.fit(o_X_train, o_y_train)
    stop = time.perf_counter()

    # Plotting
    rfrRFE_m_plot = plt.figure()
    plt.scatter(y_test, y_predicted_optimal, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Random Forest Regression with Recursive Feature Elimination - implemented manually\nOptimal Features: {}\nTime to fit the model (in seconds): {:0.4}\nNormalized RMSE: {}'.format(numFeatures-features_dropped_optimal, stop-start, optimal_rmse))
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    rfrRFE_m_plot.set_size_inches(10, 10)
    rfrRFE_m_plot.savefig('Graphs/RFR_RFE_M.png', dpi=600)
