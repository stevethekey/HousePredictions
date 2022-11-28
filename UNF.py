import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv("cleaned.csv")
    X = data.copy()
    del X['SalePrice']
    Y = data['SalePrice']
    #f_val, p_val = f_regression(X_train,y_train)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)
    ####this code was obtained from 
    # https://towardsdatascience.com/application-of-feature-selection-techniques-in-a-regression-problem-4278e2efd503
    f_val, p_val = f_regression(X_train,y_train)
    # creating a dictionary from the arrays
    feature_dict={'features':X_train.columns.tolist(),
              'f_score':f_val.tolist()}
    # creating a sorted dataframe from the dictionary
    feature_df = pd.DataFrame(feature_dict).sort_values(by='f_scores', ascending=False).reset_index(drop=True)
    # printing 25 features with the highest scores
    feature_df.iloc[:25,:]['columns'].tolist()