import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # set data = the csv file
    data = pd.read_csv('dupAmesHousing.csv')

    data.drop(['PID'], axis=1, inplace=True)

    # Write the file
    data.to_csv('benTest.csv', index=False)
