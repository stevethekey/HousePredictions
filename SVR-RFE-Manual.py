"""
Support Vector Regression with recursive feature elimination
except the RFE is partial built ins and implemented manually
"""
import copy
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # open the cleaned data
    data = pd.read_csv('cleaned.csv')

    # Create train and test files
    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    # Recursive feature elimination
    ranking_ordered = list()
    RFEsvr = SVR(kernel='linear')
    rfe = RFE(RFEsvr, n_features_to_select=1)
    rfe.fit(X_train, y_train)
    featureCols = features.columns.tolist()
    for x, y in (sorted(zip(rfe.ranking_, featureCols), key=itemgetter(0))):
        ranking_ordered.append(y)
    print(ranking_ordered)

    y_predicted_optimal = list()
    features_dropped_optimal = 1
    threshold = 0.001
    features_dropped=0
    base_rmse = 0
    while len(ranking_ordered) > 1:
        svr = SVR(kernel='linear')
        features = features.drop(ranking_ordered.pop(), axis=1)
        average_normalized_error = 0
        features_dropped += 1
        for _ in range(10):
            X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                                random_state=42)
            svr.fit(X_train, y_train)
            y_predicted = svr.predict(X_test)

            # Normalized error
            error = sqrt(mean_squared_error(y_test, y_predicted))
            normalized_error = error/(max(y_test) - min(y_test))
            average_normalized_error += normalized_error
        print('Average Normalized RMSE with {} features dropped: {}'.format(features_dropped, average_normalized_error/10))
        if features_dropped == 1:
            base_rmse = average_normalized_error/10
        if average_normalized_error/10 < base_rmse+threshold:
            y_predicted_optimal = copy.deepcopy(y_predicted)
            features_dropped_optimal = features_dropped

    print("Optimal features dropped: {}".format(features_dropped_optimal))
    # Plotting
    plt.scatter(y_test, y_predicted_optimal, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Support Vector Regression with Recursive Feature Elimination - implemented manually')
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    plt.show()
