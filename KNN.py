"""
K-Nearest Neighbor without any feature selection
This file is DEFUNCT - we do not use it in our report
"""
import numpy as np
import pandas as pd
import warnings

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # open the cleaned data
    data = pd.read_csv('cleaned.csv')

    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    neighbors = 3
    norm_errors = np.ones(20)
    while neighbors < 20:
        knn = KNeighborsRegressor(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_test)
        error = sqrt(mean_squared_error(y_test, y_predicted))
        normalized_error = error / (max(y_test) - min(y_test))
        norm_errors[neighbors] = normalized_error
        neighbors += 1

    optimal_neighbors = norm_errors.argmin()
    knn = KNeighborsRegressor(n_neighbors=optimal_neighbors)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)

    # Normalized error
    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error / (max(y_test) - min(y_test))
    print('K-Nearest Neighbor is most optimal with {} neighbors and has a normalized error of {}'.format(
        optimal_neighbors, normalized_error))

    # Plotting
    plt.scatter(y_test, y_predicted, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('K-Nearest Neighbor with no feature selection')
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    plt.show()
