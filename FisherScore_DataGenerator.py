import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfeature
from skfeature.function.similarity_based import fisher_score 
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    data = pd.read_csv('cleaned.csv')
    dftop75 = data[['Screen Porch', 'Mas Vnr Type', 'Total Bsmt SF', 'Mas Vnr Area', 'Lot Area', 'Functional',
    'BsmtFin Type 1', 'Full Bath', 'Sale Condition', 'Garage Qual', 'Half Bath', 'Bedroom AbvGr', 'Yr Sold', 'Overall Cond', 'Wood Deck SF', 
    'Paved Drive', 'MS SubClass', 'Bsmt Full Bath','SalePrice']].copy()
    dftop75.to_csv('UpperQuartile_and_above.csv')
    dfmedian = data[['Screen Porch', 'Mas Vnr Type', 'Total Bsmt SF', 'Mas Vnr Area', 'Lot Area', 'Functional',
    'BsmtFin Type 1', 'Full Bath', 'Sale Condition', 'Garage Qual', 'Half Bath', 'Bedroom AbvGr', 'Yr Sold', 'Overall Cond', 'Wood Deck SF', 
    'Paved Drive', 'MS SubClass', 'Bsmt Full Bath', 'Lot Config', 'Condition 1', 'Central Air', 'Bsmt Half Bath', 'Fireplaces', 'Electrical', 'Land Slope', 
    'Lot Frontage', 'Garage Finish', 'Roof Matl', 'Alley', 'Heating', 'Year Built', 'BsmtFin SF 1', 'Bldg Type', 'Pool Area', 'Bsmt Cond', 'Land Contour', 'SalePrice']].copy()
    dfmedian.to_csv('Median_and_above.csv')
    dftop25 = data[['Screen Porch', 'Mas Vnr Type', 'Total Bsmt SF', 'Mas Vnr Area', 'Lot Area', 'Functional',
    'BsmtFin Type 1', 'Full Bath', 'Sale Condition', 'Garage Qual', 'Half Bath', 'Bedroom AbvGr', 'Yr Sold', 'Overall Cond', 'Wood Deck SF', 
    'Paved Drive', 'MS SubClass', 'Bsmt Full Bath', 'Lot Config', 'Condition 1', 'Central Air', 'Bsmt Half Bath', 'Fireplaces', 'Electrical', 'Land Slope', 
    'Lot Frontage', 'Garage Finish', 'Roof Matl', 'Alley', 'Heating', 'Year Built', 'BsmtFin SF 1', 'Bldg Type', 'Pool Area', 'Bsmt Cond', 'Land Contour', 'Exterior 1st', 'Garage Type', 
    'Year Remod/Add', 'Low Qual Fin SF', 'Exter Qual', 'Garage Yr Blt', 'MS Zoning', 'Gr Liv Area', 'Sale Type', 'Utilities', 'Bsmt Exposure', '2nd Flr SF', 'Garage Cars', 'Condition 2', 
    'Enclosed Porch', 'Overall Qual', 'Kitchen Qual', 'Heating QC', 'SalePrice']].copy()
    print(dftop25)
    dftop25.to_csv('LowerQuartile_and_above.csv')