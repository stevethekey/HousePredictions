import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfeature
from skfeature.function.similarity_based import fisher_score 
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
  
'''
This file does the following 
1) Get The fisher Score
2) Make Plots and export them as images for the 
    * Features with a fisherscore greater than or equal to the median
    * Features with a fisherscore in the top 25% or above
    * Features with a fisherscore in the top 75% or above 
3) Once we have those plots in a seperate file dataframes will be greated
from the median, top 25 and top 75
'''

if __name__ == "__main__":
    data = pd.read_csv('cleaned.csv')
    array = data.values
    X= array[:, 0:(len(data.columns)-1)]
    Y = array[:,(len(data.columns)-1)]
    score = fisher_score.fisher_score(X, Y)
    feat_importances = pd.Series(score, data.columns[0:len(data.columns)-1])
    median = feat_importances.median()
    twenty_five = feat_importances.quantile(0.25)
    seventy_five = feat_importances.quantile(0.75)
    median_count =0
    twenty_five_count =0
    seventy_five_count =0
    for i in score:
        if (score[i] >= median):
            median_count+=1
        if (score[i] >= twenty_five):
            twenty_five_count+=1
        if (score[i] >= seventy_five):
            seventy_five_count+=1
    df_median = feat_importances[score].astype(float).nlargest(median_count)
    df_twentyfive = feat_importances[score].astype(float).nlargest(twenty_five_count)
    df_seventyfive = feat_importances[score].astype(float).nlargest(seventy_five_count)
    df_median.plot(kind = 'barh', color = 'teal')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('Features_ge_median.png', dpi=600)
    df_seventyfive.plot(kind = 'barh', color='teal')
    f = plt.gcf()
    f.set_size_inches(10,10)
    f.savefig('Features_top75.png')
    df_twentyfive.plot(kind = 'barh', color = 'teal')
    f=plt.gcf()
    f.set_size_inches(10,10)
    f.savefig('Features_top25.png')
    '''  
    This is old code that will help me for now eventually to be deleted  

    dftop20_train = data[['Condition 1', 'Lot Config', 'Bsmt Full Bath', 'MS SubClass', 'Paved Drive', 'Wood Deck SF',
    'Overall Cond', 'Yr Sold', 'Bedroom AbvGr', 'Half Bath', 'Garage Qual', 'Sale Condition', 'Full Bath', 'BsmtFin Type 1', 'Functional', 
    'Lot Area', 'Mas Vnr Area', 'Total Bsmt SF', 'Mas Vnr Type', 'Screen Porch', 'SalePrice']].copy()
    y_train = dftop20_train[['SalePrice']].copy()
    dftop20_test = data[['Condition 1', 'Lot Config', 'Bsmt Full Bath', 'MS SubClass', 'Paved Drive', 'Wood Deck SF',
    'Overall Cond', 'Yr Sold', 'Bedroom AbvGr', 'Half Bath', 'Garage Qual', 'Sale Condition', 'Full Bath', 'BsmtFin Type 1', 'Functional', 
    'Lot Area', 'Mas Vnr Area', 'Total Bsmt SF', 'Mas Vnr Type', 'Screen Porch']].copy()
    dftop20_train.to_csv('train_20.csv', index=False)
    dftop20_test.to_csv('test_20.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    df.plot(kind = 'barh', color = 'teal')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('top20_features_fisher.png', dpi=600)

    '''
    
  




    
