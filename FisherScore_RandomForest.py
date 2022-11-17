##using the data from running the Fisher Score 
##we will now traing a Random Forest 
##this has not been run yet
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    train = pd.read_csv('train_20.csv')
    test = pd.read_csv('test_20.csv')
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.25)
    my_model = RandomForestClassifier(n_estimators = 100, random_state=42, max_depth = 6)
    my_model.fit(X_train, y_train)

    print("No syntax errors")
    y_pred = my_model.predict(X_test)

    plt.scatter(y_pred, y_test)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')

    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('FisherScore_RandomForest_Plot.png', dpi = 600)
    plt.show()
    
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred, normalize = False))


