"""
Random Forest Regressor without any feature selection
"""
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor
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

    rfr = RandomForestRegressor(random_state=0)
    rfr.fit(X_train, y_train)
    y_predicted = rfr.predict(X_test)

    # Normalized error
    error = sqrt(mean_squared_error(y_test, y_predicted))
    normalized_error = error/(max(y_test) - min(y_test))
    print(normalized_error)

    plt.scatter(y_test, y_predicted, color='blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.title('Random Forest Regression with no feature selection')
    plt.xlabel('Actual Sales Price')
    plt.ylabel('Predicted Sales Price')
    plt.show()
