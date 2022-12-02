import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import math
from sklearn import metrics 
import time

'''
Since feature selection must be perfomed after 
train_test_split this grabs the intersection of X_train and the 
desired number of features
'''
def intersection_top_pct(test, train, data):
    
    list_top90 = data['features'].to_string(index=False) 
    newTest = test.copy(deep=True) 
    newTrain = train.copy(deep=True)

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

def unf():
    # reading and splitting data to train and test


    data = pd.read_csv("cleaned.csv")
    X = data.copy()
    del X['SalePrice']
    Y = data['SalePrice']
    my_model = RandomForestRegressor(max_depth = 6)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)

    #lines 42-47 were obtained from 
    #https://towardsdatascience.com/application-of-feature-selection-techniques-in-a-regression-problem-4278e2efd503

    print ("Generating top 20 percent of Features")
    f_val, p_val = f_regression(X_train,y_train)
    feature_dict={'features':X.columns.tolist(),
              'f_score':f_val.tolist()}
    feature_df = pd.DataFrame(feature_dict).sort_values(by='f_score', ascending=False).reset_index(drop=True)
    ##table to be used in reporting stage
    
   
    df_X_train = pd.DataFrame(X_train, columns=X.columns) 
    df_X_test = pd.DataFrame(X_test, columns=X.columns)


    percent90 = (math.floor(len(feature_df)*.9))
    percent80 = (math.floor(len(feature_df)*.8))
    percent70 = (math.floor(len(feature_df)*.7))
    percent60 = (math.floor(len(feature_df)*.6))
    percent50 = (math.floor(len(feature_df)*.5))
    percent40 = (math.floor(len(feature_df)*.4))
    percent30 = (math.floor(len(feature_df)*.3))
    percent20 = (math.floor(len(feature_df)*.2))
    percent10 = (math.floor(len(feature_df)*.1))

    '''
    Creating dataframes with 90, 80, 70, 60
    50, 40, 30, 20, 10
    '''

    df_top90 = feature_df.iloc[0:percent90]
    df_top80 = feature_df.iloc[0:percent80]
    df_top70 = feature_df.iloc[0:percent70]
    df_top60 = feature_df.iloc[0:percent60]
    df_top50 = feature_df.iloc[0:percent50]
    df_top40 = feature_df.iloc[0:percent40]
    df_top30 = feature_df.iloc[0:percent30]
    df_top20 = feature_df.iloc[0:percent20]
    df_top10 = feature_df.iloc[0:percent10]

    newTest_90, newTrain_90 = intersection_top_pct(df_X_test, df_X_train, df_top90)
    newTest_80, newTrain_80 = intersection_top_pct(df_X_test, df_X_train, df_top80)
    newTest_70, newTrain_70 = intersection_top_pct(df_X_test, df_X_train, df_top70)
    newTest_60, newTrain_60 = intersection_top_pct(df_X_test, df_X_train, df_top60)
    newTest_50, newTrain_50 = intersection_top_pct(df_X_test, df_X_train, df_top50)
    newTest_40, newTrain_40 = intersection_top_pct(df_X_test, df_X_train, df_top40)
    newTest_30, newTrain_30 = intersection_top_pct(df_X_test, df_X_train, df_top30)
    newTest_20, newTrain_20 = intersection_top_pct(df_X_test, df_X_train, df_top20)
    newTest_10, newTrain_10 = intersection_top_pct(df_X_test, df_X_train, df_top10)
    '''
    Exports a png with the top 20 percent of features 
    A table corresponding to them is provided in the report
    '''
    print("Generating graph for top 20 percent of features")
    df_top20 = df_top20.reset_index(drop=True)
    df_top20.reset_index(drop=True, inplace=True)
    df_top20.plot(kind = 'barh', color='blue')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('Graphs/Top20percentFeatures.png', dpi=600)

    print("Generating box plot for features importance")
    feature_df.to_csv("UNF_selection.csv")
    fig = plt.figure()
    boxplot = feature_df.boxplot(column = 'f_score')
    fig.savefig("Graphs/BoxPlotforF_score.png")
    
    normalized_root_mean_squared_error_base = 0
    normalized_root_mean_squared_error_feat90 = 0
    normalized_root_mean_squared_error_feat80 = 0
    normalized_root_mean_squared_error_feat70 = 0
    normalized_root_mean_squared_error_feat60 = 0
    normalized_root_mean_squared_error_feat50 = 0
    normalized_root_mean_squared_error_feat40 = 0
    normalized_root_mean_squared_error_feat30 = 0
    normalized_root_mean_squared_error_feat20 = 0
    normalized_root_mean_squared_error_feat10 = 0

    '''
    The following will train the RFR at varying different numbers of 
    features
    '''
    print ("Random Forest Regressor - beginning to be trained")
    RMSE_RFR = np.zeros(10)
    print ("Training the base RFR model (no Feature selection)")
    tbt_rfr_base = time.time()
    my_model.fit(X_train, y_train)
    tbt_rfr_end = time.time()
    y_pred = my_model.predict(X_test)
    normalized_root_mean_squared_error_base = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_RFR[0] = normalized_root_mean_squared_error_base
    print ("Root Mean Squared Error (Normalized)", normalized_root_mean_squared_error_base)
    
    print ("Generating Graph for no Feature Selection")
    fig1 = plt.figure('Base Feature Selection')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Base Feature Selection RFR\nRoot Mean Squared Error (Normalized): {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_base, tbt_rfr_end - tbt_rfr_base))
    fig1.set_size_inches(10, 10)
    fig1.savefig("Graphs/UNF_BASE_RFR.png", dpi = 600)
    
    print ("Training RFR with 90 percent of features")
    my_model.fit(newTrain_90.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_90.to_numpy())
    normalized_root_mean_squared_error_feat90= math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[1] = normalized_root_mean_squared_error_feat90
    if (abs(normalized_root_mean_squared_error_feat90-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error (Normalized) of 90 percent of features(Normalized)",normalized_root_mean_squared_error_feat90)

    print ("Training RFR with 80 percent of features")
    my_model.fit(newTrain_80.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_80.to_numpy())
    normalized_root_mean_squared_error_feat80 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[2] = normalized_root_mean_squared_error_feat80
    if (abs(normalized_root_mean_squared_error_feat80-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 80 percent of features (Normalized)",normalized_root_mean_squared_error_feat80)

    print ("Training RFR with 70 percent of features")
    my_model.fit(newTrain_70.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_70.to_numpy())
    normalized_root_mean_squared_error_feat70 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[3] =  normalized_root_mean_squared_error_feat70
    if (abs(normalized_root_mean_squared_error_feat70-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 70 percent of features (Normalized)",normalized_root_mean_squared_error_feat70)

    print ("Training RFE with 60 percent of the features")
    my_model.fit(newTrain_60.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_60.to_numpy())
    normalized_root_mean_squared_error_feat60 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[4] = normalized_root_mean_squared_error_feat60
    if (abs(normalized_root_mean_squared_error_feat60-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 60 percent of features (Normalized)",normalized_root_mean_squared_error_feat60)
    
    print ("Training RFE with 50 percent of the features")
    my_model.fit(newTrain_50.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_50.to_numpy())
    normalized_root_mean_squared_error_feat50 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[5] = normalized_root_mean_squared_error_feat50
    if (abs(normalized_root_mean_squared_error_feat50-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 50 percent of features (Normalized)",normalized_root_mean_squared_error_feat50)

    print ("Training RFE with 40 percent of the features")
    tbt_rfr_optimal = time.time()
    my_model.fit(newTrain_40.to_numpy(), y_train)
    tbt_rfr_optimal_end = time.time()
    y_pred = my_model.predict(newTest_40.to_numpy())
    normalized_root_mean_squared_error_feat40 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[6] = normalized_root_mean_squared_error_feat40
    if (abs(normalized_root_mean_squared_error_feat40-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 40 percent of features (Normalized)",normalized_root_mean_squared_error_feat40)
    

    print ("Plotting the optimal number of features")
    fig2 = plt.figure('Top 40 percent Feature Selection')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Top 40 percent Feature Selection RFR\n Root Mean Squared Error (Normalized): {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_feat40, tbt_rfr_optimal_end - tbt_rfr_optimal))
    fig2.set_size_inches(10, 10)
    fig2.savefig("Graphs/UNF_optimal_RFR.png", dpi = 600)

    print ("Training RFR with 30 percent of features")
    my_model.fit(newTrain_30.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_30.to_numpy())
    normalized_root_mean_squared_error_feat30 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[7] = normalized_root_mean_squared_error_feat30
    if (abs(normalized_root_mean_squared_error_feat30-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 30 percent of features (Normalized)",normalized_root_mean_squared_error_feat30)

    print ("Training RFR with 20 percent of features")
    my_model.fit(newTrain_20.to_numpy(), y_train)
    y_pred = my_model.predict(newTest_20.to_numpy())
    normalized_root_mean_squared_error_feat20 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[8] = normalized_root_mean_squared_error_feat20
    if (abs(normalized_root_mean_squared_error_feat20-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 20 percent of features (Normalized)",normalized_root_mean_squared_error_feat20)

    print ("Training RFR with 10 percent of features")
    tbt_rfr_top10 =time.time()
    my_model.fit(newTrain_10.to_numpy(), y_train)
    tbt_rfr_top_end = time.time()
    y_pred = my_model.predict(newTest_10.to_numpy())
    normalized_root_mean_squared_error_feat10 = math.sqrt(metrics.mean_squared_error(y_test, y_pred)) / ((y_test.max() - y_test.min()))
    RMSE_RFR[9] = normalized_root_mean_squared_error_feat10
    if (abs(normalized_root_mean_squared_error_feat10-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error of top 10 percent of features (Normalized)",normalized_root_mean_squared_error_feat10)

    print ("Graphing RFR with 10 percent of the features")
    fig3 = plt.figure('Top 10 percent features')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Top 10 Percent Feature Selection RFR\nRoot Mean Square Error (Normalized) {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_feat10,tbt_rfr_top_end-tbt_rfr_top10))
    fig3.set_size_inches(10, 10)
    fig3.savefig("Graphs/UNF_10_RFR.png", dpi = 600)

    print ("Plotting the RMSE over multiple percentages of features for RFR")
    RMSE_df_RFR = pd.DataFrame(RMSE_RFR)
    RMSE_df_RFR.plot(kind = 'line', color = 'blue')
    fig_RMSERFR = plt.gcf()
    plt.title("Normalized RMSE over Percentage of Features Dropped")
    plt.xlabel("(Percentage of Features dropped)* 100")
    plt.ylabel("RMSE (Normalized) ")
    fig_RMSERFR.set_size_inches(10, 10)
    fig_RMSERFR.savefig('Graphs/RMSE_RFR.png', dpi = 600)

    print ("Support Vector Regressor")
    RMSE_SVR = np.zeros(10)

    '''
    The following will train the SVR with various percentage of features
    '''
    print("Beginning to train SVR with no Feature Selection")
    tbt_svc_base = time.time()
    my_model2 = SVR(kernel = 'linear')
    my_model2.fit(X_train, y_train)
    tbt_svc_base_end = time.time()
    y_pred = my_model2.predict(X_test)
    normalized_root_mean_squared_error_base = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[0] = normalized_root_mean_squared_error_base
    print ("Average Root Mean Squared Error for SVC (Normalized)",normalized_root_mean_squared_error_base)

    print("Beginning to make graph for SVR with no Feature Selection")
    figSVRbase = plt.figure('Base Feature Selection SVR')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Base Feature Selection SVR\nRoot Mean Squared Error (Normalized): {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_base,  tbt_svc_base_end-tbt_svc_base))
    figSVRbase.set_size_inches(10, 10)
    figSVRbase.savefig("Graphs/UNF_BASE_SVR.png", dpi = 600)

    print("Training SVC with 90 percent of features")
    my_model2.fit(newTrain_90.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_90.to_numpy())
    normalized_root_mean_squared_error_feat90 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[1] = normalized_root_mean_squared_error_feat90
    if (abs(normalized_root_mean_squared_error_feat90-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 90 percent of features",normalized_root_mean_squared_error_feat90)

    print("Training SVC with 80 percent of features")
    my_model2.fit(newTrain_80.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_80.to_numpy())
    normalized_root_mean_squared_error_feat80 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[2] = normalized_root_mean_squared_error_feat80
    if (abs(normalized_root_mean_squared_error_feat80-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 80 percent of features",normalized_root_mean_squared_error_feat80)

    print("Training SVC with 70 percent of features")
    my_model2.fit(newTrain_70.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_70.to_numpy())
    normalized_root_mean_squared_error_feat70 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[3] = normalized_root_mean_squared_error_feat70
    if (abs(normalized_root_mean_squared_error_feat70-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 70 percent of features",normalized_root_mean_squared_error_feat70)

    print("Training SVC with 60 percent of features")
    my_model2.fit(newTrain_60.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_60.to_numpy())
    normalized_root_mean_squared_error_feat60 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[4] = normalized_root_mean_squared_error_feat60
    if (abs(normalized_root_mean_squared_error_feat60-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 60 percent of features",normalized_root_mean_squared_error_feat60)

    print("Training SVC with 50 percent of features")
    my_model2.fit(newTrain_50.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_50.to_numpy())
    normalized_root_mean_squared_error_feat50 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[5] = normalized_root_mean_squared_error_feat50
    if (abs(normalized_root_mean_squared_error_feat50-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 50 percent of features",normalized_root_mean_squared_error_feat50)

    print("Training SVC with 40 percent of features")
    my_model2.fit(newTrain_40.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_40.to_numpy())
    normalized_root_mean_squared_error_feat40 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[6] = normalized_root_mean_squared_error_feat40
    if (abs(normalized_root_mean_squared_error_feat40-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 40 percent of features",normalized_root_mean_squared_error_feat40)

    print("Training SVC with 30 percent of features")
    tbt_svc_optimal = time.time()
    my_model2.fit(newTrain_30.to_numpy(), y_train)
    tbt_svc_otpimal_end = time.time()
    y_pred = my_model2.predict(newTest_30.to_numpy())
    normalized_root_mean_squared_error_feat30 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[7] = normalized_root_mean_squared_error_feat30
    if (abs(normalized_root_mean_squared_error_feat30-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 30 percent of features",normalized_root_mean_squared_error_feat30)

    print ("Plotting SVC with 30 percent of the features")
    fig4 =  plt.figure('Top 30 percent Feature Selection')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Base Feature Selection SVR\nRoot Mean Squared Error (Normalized): {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_feat30,  tbt_svc_otpimal_end-tbt_svc_optimal))
    fig4.set_size_inches(10, 10)
    fig4.savefig("Graphs/UNF_SVR_optimal.png", dpi = 600)

    print ("Training SVC with 20 percent of the features")
    my_model2.fit(newTrain_20.to_numpy(), y_train)
    y_pred = my_model2.predict(newTest_20.to_numpy())
    normalized_root_mean_squared_error_feat20 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[8] = normalized_root_mean_squared_error_feat20
    if (abs(normalized_root_mean_squared_error_feat20-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 20 percent of features",normalized_root_mean_squared_error_feat20)

    print ("Training SVC with 10 percent of the features")
    tbt_svc_top10 = time.time()
    my_model2.fit(newTrain_10.to_numpy(), y_train)
    tbt_svc_top10_end = time.time()
    y_pred = my_model2.predict(newTest_10.to_numpy())
    normalized_root_mean_squared_error_feat10 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))/ ((y_test.max()-y_test.min()))
    RMSE_SVR[9] = normalized_root_mean_squared_error_feat10
    if (abs(normalized_root_mean_squared_error_feat10-normalized_root_mean_squared_error_base) <= 0.001):
        print ("Optimal")
    print ("Average Root Mean Squared Error for top 10 percent of features",normalized_root_mean_squared_error_feat10)
 
    print ("Plotting SVC with 10 percent of the features")
    fig5 =  plt.figure('Top 10 percent Feature Selection')
    plt.scatter(y_test, y_pred, color = 'blue')
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Base Feature Selection SVR\nRoot Mean Squared Error (Normalized): {}\nTraining Time: {}'.format(normalized_root_mean_squared_error_feat10, tbt_svc_top10_end -tbt_svc_top10))
    fig5.set_size_inches(10, 10)
    fig5.savefig("Graphs/UNF_top10_SVR.png", dpi = 600)

    print ("Plotting RMSE plotting the RMSE over multiple percentages of features for SVR")
    RMSE_df_SVR = pd.DataFrame(RMSE_SVR)
    RMSE_df_SVR.plot(kind = 'line', color = 'blue')
    fig_RMSESVR = plt.gcf()
    plt.title("Normalized RMSE over Percentage of Features Dropped")
    plt.xlabel("(Percentage of Features dropped) * 100")
    plt.ylabel("RMSE (Normalized) ")
    fig_RMSESVR.set_size_inches(10, 10)
    fig_RMSESVR.savefig('Graphs/RMSE_SVR.png', dpi = 600)