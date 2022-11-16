import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfeature
from skfeature.function.similarity_based import fisher_score 
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

#using the cleaned data 
#first we must split into a training dataset and a test dataset 
#then we will extract the top 20 features 
#then train the random forest   
if __name__ == "__main__":

    train = pd.read_csv('cleaned.csv')
    test = train
    test.drop(['SalePrice'], axis=1, inplace=True)

    # open the cleaned data
    data = pd.read_csv('cleaned.csv')
    array = data.values
    X= array[:, 0:(len(data.columns)-1)]
    Y = array[:,(len(data.columns)-1)]

    #gets the fisher score for each important feature
    score = fisher_score.fisher_score(X, Y)

    feat_importances = pd.Series(score, data.columns[0:len(data.columns)-1])
    df = feat_importances[score].astype(float).nlargest(20)
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
    
    
  




    
