"""
Support Vector Regression with Recursive Feature Elimination
implemented using built-ins
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # Open the cleaned data
    data = pd.read_csv('cleaned.csv')

    # Prep the train / test files
    features = data.drop('SalePrice', axis=1)
    target = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.33,
                                                        random_state=42)

    # RFE cross validation
    rfecv = RFECV(estimator=SVR(kernel='linear'), min_features_to_select=1, n_jobs=-1, scoring="r2")
    rfecv.fit(X_train, y_train)
    y_predicted = rfecv.predict(X_test)
    features_dropped = len(features.columns) - len(features.columns[rfecv.support_])
    # print(features.columns[rfecv.support_])

    # Normalized error
    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error / (max(y_test) - min(y_test))
    # print('SVR with RFE: Normalized RMSE with {} features dropped: {}'.format(features_dropped, normalized_error))

    # Plotting
    svrRFE_plot = plt.figure()
    plt.scatter(y_test, y_predicted, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Support Vector Regression with Recursive Feature Elimination\nOptimal Features: {}\nNormalized RMSE: {}'.format(len(features.columns)-features_dropped, normalized_error))
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    svrRFE_plot.set_size_inches(10, 10)
    svrRFE_plot.savefig('Graphs/SVR_RFE.png', dpi=600)
