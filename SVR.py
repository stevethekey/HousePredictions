"""
Support Vector Regression with no feature selection
"""
import time

import numpy as np
import pandas as pd
import warnings
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


def svr():
    # Open the cleaned data
    data = pd.read_csv('cleaned.csv')

    # Prep train / test files
    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    # SVR
    svrModel = SVR(kernel='linear')
    start = time.perf_counter()
    svrModel.fit(X_train, y_train)
    stop = time.perf_counter()
    y_predicted = svrModel.predict(X_test)

    # Normalized error
    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error / (max(y_test) - min(y_test))
    # print('Normalized error of Support Vector Regressor without feature selection: {}'.format(normalized_error))

    # Plotting
    svr_plot = plt.figure()
    plt.scatter(y_test, y_predicted, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Support Vector Regression with no feature selection\nFeatures: {}\nTime to fit the model (in seconds): {:0.4}\nNormalized RMSE: {}'.format(len(features.columns), stop-start, normalized_error))
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    svr_plot.set_size_inches(10, 10)
    svr_plot.savefig('Graphs/SVR_BASE.png', dpi=600)
