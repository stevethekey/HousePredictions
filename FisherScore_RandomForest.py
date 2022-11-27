#using the data from running the Fisher Score 
#we will now traing a Random Forest 
#this has not been run yet
import numpy as np
import pandas as pd
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn import metrics 
from FisherScore_DataGenerator import GenData

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":

    ###this is the baseline 
    data = pd.read_csv("cleaned.csv")
    X = data.copy()

    del X['SalePrice']
    Y = data['SalePrice']
    my_model = RandomForestRegressor(n_estimators = 300, max_depth = 6)
    X_train, X_test, y_train, y_test = train_test_split (X.to_numpy(), Y.to_numpy(), test_size = 0.33, random_state =42)
    #once the data is split perform feature selection on the training sets
    X_train_df  = pd.DataFrame(X_train, columns = X.columns)
    X_test_df = pd.DataFrame (X_test, columns = X.columns)
    X_train_df_feat = X_train_df[['Screen Porch', 'Mas Vnr Type', 'Total Bsmt SF', 'Mas Vnr Area', 'Lot Area', 'Functional',
    'BsmtFin Type 1', 'Full Bath', 'Sale Condition', 'Garage Qual', 'Half Bath', 'Bedroom AbvGr', 'Yr Sold', 'Overall Cond', 'Wood Deck SF', 
    'Paved Drive', 'MS SubClass', 'Bsmt Full Bath', 'Lot Config', 'Condition 1', 'Central Air', 'Bsmt Half Bath', 'Fireplaces', 'Electrical', 'Land Slope', 
    'Lot Frontage', 'Garage Finish', 'Roof Matl', 'Alley', 'Heating', 'Year Built', 'BsmtFin SF 1', 'Bldg Type', 'Pool Area', 'Bsmt Cond', 'Land Contour', 'Exterior 1st', 'Garage Type', 
    'Year Remod/Add', 'Low Qual Fin SF', 'Exter Qual', 'Garage Yr Blt', 'MS Zoning', 'Gr Liv Area', 'Sale Type', 'Utilities', 'Bsmt Exposure', '2nd Flr SF', 'Garage Cars', 'Condition 2', 
    'Enclosed Porch', 'Overall Qual', 'Kitchen Qual', 'Heating QC', 'House Style', 'Neighborhood', 'Lot Shape', 'Exter Cond', 'Foundation', 'Bsmt Unf SF', 'BsmtFin Type 2']].copy()
    X_test_df_feat = X_test_df[['Screen Porch', 'Mas Vnr Type', 'Total Bsmt SF', 'Mas Vnr Area', 'Lot Area', 'Functional',
    'BsmtFin Type 1', 'Full Bath', 'Sale Condition', 'Garage Qual', 'Half Bath', 'Bedroom AbvGr', 'Yr Sold', 'Overall Cond', 'Wood Deck SF', 
    'Paved Drive', 'MS SubClass', 'Bsmt Full Bath', 'Lot Config', 'Condition 1', 'Central Air', 'Bsmt Half Bath', 'Fireplaces', 'Electrical', 'Land Slope', 
    'Lot Frontage', 'Garage Finish', 'Roof Matl', 'Alley', 'Heating', 'Year Built', 'BsmtFin SF 1', 'Bldg Type', 'Pool Area', 'Bsmt Cond', 'Land Contour', 'Exterior 1st', 'Garage Type', 
    'Year Remod/Add', 'Low Qual Fin SF', 'Exter Qual', 'Garage Yr Blt', 'MS Zoning', 'Gr Liv Area', 'Sale Type', 'Utilities', 'Bsmt Exposure', '2nd Flr SF', 'Garage Cars', 'Condition 2', 
    'Enclosed Porch', 'Overall Qual', 'Kitchen Qual', 'Heating QC', 'House Style', 'Neighborhood', 'Lot Shape', 'Exter Cond', 'Foundation', 'Bsmt Unf SF', 'BsmtFin Type 2']].copy()
    X_train_to_use = X_train_df_feat.to_numpy()
    X_test_to_use = X_test_df_feat.to_numpy()
    my_model.fit(X_train_to_use, y_train)
    y_pred = my_model.predict(X_test_to_use)
    print (metrics.r2_score(y_test, y_pred))
    




    '''
    print (X_train)
    my_model.fit(X_train, y_train)
    y_pred = my_model.predict(X_test)
    print (metrics.r2_score(y_test, y_pred))
    root_mean_squared_error = (math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    normalized_root_mean_squared_error = root_mean_squared_error/(y_test.max()-y_test.min())
    print ("Root Mean Squared Error", normalized_root_mean_squared_error)
    list_of_i_for_top_seventyfive = []
    list_of_i_for_median = []
    list_of_i_for_top_twenty_five = []
    list_of_i_for_top_seventyfive,  list_of_i_for_median, list_of_i_for_top_twenty_five = GenData()
    #Now lets try it on the top 25% of features 
    #
    #for i in range (len(list_of_i_for_top_seventyfive)-1):
     #   X_test_to_use[i] = 
    
  #  print (X_train_to_use)

    '''
    '''
    Old code that will eventually be the deleted but for now is helping me 
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
        print (Y)
        
        X_to_use = X.to_numpy()
        Y_to_use = Y.to_numpy()
        my_model = RandomForestRegressor(n_estimators = 100, random_state=4, max_depth = 6)
        X_train, X_test, y_train, y_test = train_test_split(X_to_use, Y_to_use, test_size = 0.2, random_state=4)
        my_model.fit(X_train, y_train)
        print(my_model.score(X_test, y_test))
        y_pred = my_model.predict(X_test)
        print (y_test)
        print (y_train)
        my_model_r2 = metrics.r2_score(y_test, y_pred)
        root_mean_squared_error = (math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        normalized_root_mean_squared_error = root_mean_squared_error/(y_test.max()-y_test.min())
        print ("R Squared", my_model_r2)
        print ("Root Mean Squared Error", normalized_root_mean_squared_error)
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
        '''

