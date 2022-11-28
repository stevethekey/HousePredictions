import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import math
from sklearn import metrics 
'''
Steps to be taken 
1. split to X_train and Y_train  (Good)
2. Find the optimal value of K on base model (Good)
3. Plot it (Plot K Values)
4. Then the feature selection stuff from last time 
'''
if __name__ == "__main__":
    data = pd.read_csv("cleaned.csv")
    X = data.copy()
    del X['SalePrice']
    Y = data['SalePrice']
    r2_score_for_base = np.zeros(30)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)
    for j in range (0, 3):
        for i in range (0, 30):
            my_model = KNeighborsRegressor(n_neighbors = i+1)
            my_model.fit(X_train, y_train)
            y_pred = my_model.predict(X_test)
            r2_score_for_base[i] = metrics.r2_score(y_test, y_pred)
        print("Best K Value is ", r2_score_for_base.argmax()+1)
        print ("R2 Score of best K Value is ", r2_score_for_base.max())
    

