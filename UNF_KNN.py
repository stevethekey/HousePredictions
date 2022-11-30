import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from sklearn import metrics 
from sklearn.metrics import mean_squared_error

def getOptimalK():
    data = pd.read_csv("cleaned.csv")
    X = data.copy()
    del X['SalePrice']
    Y = data['SalePrice']
    r2_score_for_base = np.zeros(30)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)

    rmse_val = np.zeros(30)

    for i in range (0, 30):
        my_model = KNeighborsRegressor(n_neighbors = i+1)
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
        r2_score_for_base[i] = metrics.r2_score(y_test, y_pred)

        error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
        rmse_val[i] = error #store rmse values

    print("Best K Value is ",rmse_val.argmin()+1)
    K = rmse_val.argmin()+1
    print ("R2 Score of best K Value is ", rmse_val.min())
    print('RMSE value (normalized) for k= ' , K , 'is:', error/(y_test.max()-y_test.min()))

   # curve = pd.DataFrame(rmse_val) #elbow curve 
   # curve.plot()
   # plt.show()
    return K