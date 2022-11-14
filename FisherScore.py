import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfeature
from skfeature.function.similarity_based import fisher_score 
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # open the cleaned data
    data = pd.read_csv('cleaned.csv')
    array = data.values
    X= array[:, 0:(len(data.columns)-1)]
    Y = array[:,(len(data.columns)-1)]
    score = fisher_score.fisher_score(X, Y)

    feat_importances = pd.Series(score, data.columns[0:len(data.columns)-1])
    df = feat_importances[score].astype(float).nlargest(20)
    df.plot(kind = 'barh', color = 'teal')
    f = plt.gcf()
    f.set_size_inches(10, 10)
    f.savefig('top20_features_fisher.png', dpi=600)
    plt.show()

    
