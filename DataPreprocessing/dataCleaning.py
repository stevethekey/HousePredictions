import inline as inline
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

if __name__ == "__main__":
    # set data = the csv file
    data = pd.read_csv('dupAmesHousing.csv')

    data.drop(['PID'], axis=1, inplace=True)

    # Write the file
    data.to_csv('benTest.csv', index=False)
