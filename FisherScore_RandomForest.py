##using the data from running the Fisher Score 
##we will now traing a Random Forest 
##this has not been run yet
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    train = pd.read_csv('train_20.csv')
    test = pd.read_csv('test_20.csv')
    X_train, X_test, y_train, y_test = train_test_split(train, test, testsize = 0.30)
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


