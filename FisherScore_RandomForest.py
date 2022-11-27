##using the data from running the Fisher Score 
##we will now traing a Random Forest 
##this has not been run yet
import numpy as np
import pandas as pd
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn import metrics 

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    for i in range (0,4,1):
        if (i ==0):
            X = pd.read_csv('cleaned.csv')
        elif (i ==1):
            X = pd.read_csv('UpperQuartile_and_above.csv')
        elif (i==2):
            X = pd.read_csv('Median_and_above.csv')
        elif (i ==3):
            X = pd.read_csv('LowerQuartile_and_above.csv')
        Y = X['SalePrice']
        X_to_use = X.to_numpy()
        Y_to_use = Y.to_numpy()
        my_model = RandomForestRegressor(n_estimators = 100, random_state=42, max_depth = 6)
        X_train, X_test, y_train, y_test = train_test_split(X_to_use, Y_to_use, test_size = 0.33, random_state=4)
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
        my_model_r2 = metrics.r2_score(y_test, y_pred)
        root_mean_squared_error = (metrics.mean_squared_error(y_test, y_pred, squared=False))
        print ("R Squared", my_model_r2)
        print ("Root Mean Squared Error", root_mean_squared_error)
        plt.scatter(y_pred, y_test)
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        diagonal = np.linspace(0, np.max(y_test), 100)
        plt.plot(diagonal, diagonal, '-r')
        f = plt.gcf()
        f.set_size_inches(10, 10)
        if (i == 0):
            f.savefig('Baseline.png', dpi = 600)
        elif (i == 1):
            f.savefig('top25pctfeat.png', dpi = 600)
        elif (i ==2):
            f.savefig('Medianandabovefeat.png', dpi = 600)
        elif (i ==3):
            f.savefig('top75pctfeat.png', dpi = 600)


