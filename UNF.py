import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import metrics 

if __name__ == "__main__":
    data = pd.read_csv("cleaned.csv")
    X = data.copy()
    del X['SalePrice']
    Y = data['SalePrice']
    my_model = RandomForestRegressor(n_estimators = 300, max_depth = 6)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)
    ####lines 17-20 were obtained from 
    # https://towardsdatascience.com/application-of-feature-selection-techniques-in-a-regression-problem-4278e2efd503
    f_val, p_val = f_regression(X_train,y_train)
    feature_dict={'features':X.columns.tolist(),
              'f_score':f_val.tolist()}
    feature_df = pd.DataFrame(feature_dict).sort_values(by='f_score', ascending=False).reset_index(drop=True)
    ##table to be used in reporting stage
    df_top20 = feature_df.iloc[0:20]
    df_top20 = df_top20.reset_index(drop=True)
    df_top20.reset_index(drop=True, inplace=True)
    print (df_top20)
    df_top20.plot(kind = 'barh', color='teal')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('Top20Features.png', dpi=600)
    feature_df.to_csv("UNF_selection.csv")
    fig = plt.figure()
    boxplot = feature_df.boxplot(column = 'f_score')
    fig.savefig("BoxPlotforF_score.png")
    r2_score_for_base = np.zeros(10)
    normalized_root_mean_squared_error_base = np.zeros(10)
    for i in range(0, 10, 1):
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
        r2_score_for_base[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_base[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))

    avg_r2_score_for_base = np.mean(r2_score_for_base)
    avg_RMSENORM_base = np.mean(normalized_root_mean_squared_error_base)
    print ("Avg R2 Score for no feature selection", avg_r2_score_for_base)
    print ("Average Root Mean Squared Error (Normalized",avg_RMSENORM_base)
    
