import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import metrics 

def intersection_top_pct(test, train, data):
    
    list_top90 = data['features'].to_string(index=False) 
    newTest = test.copy(deep=True) 
    newTrain = train.copy(deep=True);

    for i in range(len(test.columns)):
        col = test.columns[i] 
        if col not in list_top90:
            newTest.drop(labels=col, axis=1, inplace=True) 
    
    for i in range(len(train.columns)):
        col = train.columns[i] 
        if col not in list_top90:
            newTrain.drop(labels=col, axis=1, inplace=True) 

    return newTest, newTrain 

    # loop through test, if column found in data good, else drop

if __name__ == "__main__":

    # reading and splitting data to train and test
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
    
   
    df_X_train = pd.DataFrame(X_train, columns=X.columns) 
    df_X_test = pd.DataFrame(X_test, columns=X.columns)

    percent90 = (math.floor(len(feature_df)*.9))
    percent85 = (math.floor(len(feature_df)*.85))
    percent80 = (math.floor(len(feature_df)*.8))
    percent75 = (math.floor(len(feature_df)*.75))
    percent70 = (math.floor(len(feature_df)*.7))
    percent60 = (math.floor(len(feature_df)*.6))
    percent40 = (math.floor(len(feature_df)*.4))
    percent20 = (math.floor(len(feature_df)*.2))
    df_top90 = feature_df.iloc[0:percent90]
    df_top85 = feature_df.iloc[0:percent85]
    df_top80 = feature_df.iloc[0:percent80]
    df_top75 = feature_df.iloc[0:percent75]
    df_top70 = feature_df.iloc[0:percent70]
    df_top60 = feature_df.iloc[0:percent60]
    df_top40 = feature_df.iloc[0:percent40]
    df_top20 = feature_df.iloc[0:percent20]

    newTest_90, newTrain_90 = intersection_top_pct(df_X_test, df_X_train, df_top90)
    newTest_85, newTrain_85 = intersection_top_pct(df_X_test, df_X_train, df_top85)
    newTest_80, newTrain_80 = intersection_top_pct(df_X_test, df_X_train, df_top80)
    newTest_75, newTrain_75 = intersection_top_pct(df_X_test, df_X_train, df_top75)
    newTest_70, newTrain_70 = intersection_top_pct(df_X_test, df_X_train, df_top70)
    newTest_60, newTrain_60 = intersection_top_pct(df_X_test, df_X_train, df_top60)
    newTest_40, newTrain_40 = intersection_top_pct(df_X_test, df_X_train, df_top40)
    newTest_20, newTrain_20 = intersection_top_pct(df_X_test, df_X_train, df_top20)
    '''
    df_top20 = df_top20.reset_index(drop=True)
    df_top20.reset_index(drop=True, inplace=True)
    df_top20.plot(kind = 'barh', color='teal')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('Top20Features.png', dpi=600)
    feature_df.to_csv("UNF_selection.csv")
    fig = plt.figure()
    boxplot = feature_df.boxplot(column = 'f_score')
    fig.savefig("BoxPlotforF_score.png")
    '''
    r2_score_for_base = np.zeros(10)
    normalized_root_mean_squared_error_base = np.zeros(10)

    r2_score_for_feat90 = np.zeros(10)
    r2_score_for_feat85 = np.zeros(10)
    r2_score_for_feat80 = np.zeros(10)
    r2_score_for_feat75 = np.zeros(10)
    normalized_root_mean_squared_error_feat90 = np.zeros(10)
    normalized_root_mean_squared_error_feat85 = np.zeros(10)
    normalized_root_mean_squared_error_feat80 = np.zeros(10)
    normalized_root_mean_squared_error_feat75 = np.zeros(10)
    r2_score_for_feat70 = np.zeros(10)
    normalized_root_mean_squared_error_feat70 = np.zeros(10)
    r2_score_for_feat60 = np.zeros(10)
    normalized_root_mean_squared_error_feat60 = np.zeros(10)
    r2_score_for_feat40 = np.zeros(10)
    normalized_root_mean_squared_error_feat40 = np.zeros(10)
    r2_score_for_feat20 = np.zeros(10)
    normalized_root_mean_squared_error_feat20 = np.zeros(10)

    avg_r2_score_for_feat90 = 0
    avg_r2_score_for_feat85 = 0
    avg_r2_score_for_feat80 = 0
    avg_r2_score_for_feat75 =0
    avg_r2_score_for_feat70 = 0
    avg_r2_score_for_feat60 = 0
    avg_r2_score_for_feat40 = 0
    avg_r2_score_for_feat20 = 0
    avg_RMSENORM_feat90 =0
    avg_RMSENORM_feat85 = 0
    avg_RMSENORM_feat80 =0
    avg_RMSENORM_feat75 =0
    avg_RMSENORM_feat70 = 0
    avg_RMSENORM_feat60 = 0
    avg_RMSENORM_feat40 = 0
    avg_RMSENORM_feat20 = 0


    print(newTest_90)
    print(newTest_85)
    print(newTest_80)
    print(newTest_75)
    print(newTest_70)
    print(newTest_60)


    for i in range(0, 10, 1):
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
        r2_score_for_base[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_base[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    avg_r2_score_for_base = np.mean(r2_score_for_base)
    avg_RMSENORM_base = np.mean(normalized_root_mean_squared_error_base)
    print ("Avg R2 Score for no feature selection", avg_r2_score_for_base)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_base)

    for i in range(0, 10, 1):
        my_model.fit(newTrain_90.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_90.to_numpy())
        r2_score_for_feat90[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat90[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat90 = np.mean(r2_score_for_feat90)
    avg_RMSENORM_feat90 = np.mean(normalized_root_mean_squared_error_feat90)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat90)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat90)

    for i in range(0, 10, 1):
        my_model.fit(newTrain_85.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_85.to_numpy())
        r2_score_for_feat85[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat85[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat85 = np.mean(r2_score_for_feat85)
    avg_RMSENORM_feat85 = np.mean(normalized_root_mean_squared_error_feat85)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat85)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat85)


    for i in range(0, 10, 1):
        my_model.fit(newTrain_80.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_80.to_numpy())
        r2_score_for_feat80[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat80[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat80 = np.mean(r2_score_for_feat80)
    avg_RMSENORM_feat80 = np.mean(normalized_root_mean_squared_error_feat80)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat80)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat80)

    for i in range(0, 10, 1):
        my_model.fit(newTrain_75.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_75.to_numpy())
        r2_score_for_feat75[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat75[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat75 = np.mean(r2_score_for_feat75)
    avg_RMSENORM_feat75 = np.mean(normalized_root_mean_squared_error_feat75)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat75)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat75)

    
    for i in range(0, 10, 1):
        my_model.fit(newTrain_70.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_70.to_numpy())
        r2_score_for_feat70[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat70[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat70 = np.mean(r2_score_for_feat70)
    avg_RMSENORM_feat70 = np.mean(normalized_root_mean_squared_error_feat70)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat70)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat70)

    
    for i in range (0, 10, 1):
        my_model.fit(newTrain_60.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_60.to_numpy())
        r2_score_for_feat60[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat60[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat60 = np.mean(r2_score_for_feat60)
    avg_RMSENORM_feat60 = np.mean(normalized_root_mean_squared_error_feat60)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat60)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat60)
    
    for i in range (0, 10, 1):
        my_model.fit(newTrain_40.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_40.to_numpy())
        r2_score_for_feat40[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat40[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat40 = np.mean(r2_score_for_feat40)
    avg_RMSENORM_feat40 = np.mean(normalized_root_mean_squared_error_feat40)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat40)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat40)

    for i in range (0, 10, 1):
        my_model.fit(newTrain_20.to_numpy(), y_train)
        y_pred = my_model.predict(newTest_20.to_numpy())
        r2_score_for_feat20[i] = metrics.r2_score(y_test, y_pred)
        normalized_root_mean_squared_error_feat20[i] = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    avg_r2_score_for_feat20 = np.mean(r2_score_for_feat20)
    avg_RMSENORM_feat20 = np.mean(normalized_root_mean_squared_error_feat20)
    print ("Avg R2 Score for feature selection", avg_r2_score_for_feat20)
    print ("Average Root Mean Squared Error (Normalized)",avg_RMSENORM_feat20)


