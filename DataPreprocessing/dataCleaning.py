import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # set data = the csv file
<<<<<<< Updated upstream
    data = pd.read_csv('AmesHousing.csv')

    # Cleaning the Data. Documented in Dataprocessing/datacleaning.md
    data.drop(['PID', 'Order'], axis=1, inplace=True)

    """
    Ben's columns 21-40
    """
    roof_style_mapper = {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5}
    data['Roof Style'].replace(roof_style_mapper, inplace=True)

    roof_matl_mapper = {'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6,
                        'WdShngl': 7}
    data['Roof Matl'].replace(roof_matl_mapper, inplace=True)

    exterior1st_mapper = {'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4,
                          'CmntBd': 5, 'CemntBd': 5, 'CmentBd': 5,
                          'HdBoard': 6, 'ImStucc': 7, 'MetalSd': 8, 'Other': 9, 'Plywood': 10, 'PreCast': 11,
                          'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Shng': 15, 'Wd Sdng': 15, 'WdShing': 16}
    data['Exterior 1st'].replace(exterior1st_mapper, inplace=True)

    exterior2nd_mapper = {'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4,
                          'CmntBd': 5, 'CemntBd': 5, 'CmentBd': 5,
                          'HdBoard': 6, 'ImStucc': 7, 'MetalSd': 8, 'Other': 9, 'Plywood': 10, 'PreCast': 11,
                          'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Shng': 15, 'Wd Sdng': 15, 'WdShing': 16}
    data['Exterior 2nd'].replace(exterior2nd_mapper, inplace=True)

    mas_vnr_type_mapper = {'BrkCmn': 0, 'BrkFace': 1, 'CBlock': 2, 'None': 3, 'Stone': 4}
    data['Mas Vnr Type'].replace(mas_vnr_type_mapper, inplace=True)

    exter_qual_mapper = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4}
    data['Exter Qual'].replace(exter_qual_mapper, inplace=True)

    exter_cond_mapper = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4}
    data['Exter Cond'].replace(exter_cond_mapper, inplace=True)

    foundation_mapper = {'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}
    data['Foundation'].replace(foundation_mapper, inplace=True)

    bsmt_qual_mapper = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5}
    data['Bsmt Qual'].replace(bsmt_qual_mapper, inplace=True)

    bsmt_cond_mapper = {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5}
    data['Bsmt Cond'].replace(bsmt_cond_mapper, inplace=True)

    bsmt_exposure_mapper = {'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'NA': 4}
    data['Bsmt Exposure'].replace(bsmt_exposure_mapper, inplace=True)

    bsmtfin_type_1_mapper = {'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'NA': 6}
    data['BsmtFin Type 1'].replace(bsmtfin_type_1_mapper, inplace=True)

    bsmtfin_type_2_mapper = {'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'NA': 6}
    data['BsmtFin Type 2'].replace(bsmtfin_type_2_mapper, inplace=True)

    heating_mapper = {'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5}
    data['Heating'].replace(heating_mapper, inplace=True)

    """
    Gabe's columns 41-60
    """
    Dict = dict({"GasA": 0, "Wall": 1, "Grav": 2, "GasW": 3, "Floor": 4, "OthW": 5})
    data.replace({"Heating": Dict}, inplace=True)
    Dict = dict({"Fa": 0, "TA": 1, "Ex": 2, "Gd": 3, "Po": 4, "NA": 5})
    data.replace({"Heating QC": Dict}, inplace=True)
    data.replace({"Kitchen Qual": Dict}, inplace=True)
    data.replace({"Fireplace Qu": Dict}, inplace=True)
    Dict = dict({"Y": 0, "N": 1})
    data.replace({"Central Air": Dict}, inplace=True)
    Dict = dict({"SBrkr": 0, "FuseA": 1, "FuseF": 2, "FuseP": 3, "Mix": 4})
    data.replace({"Electrical": Dict}, inplace=True)
    Dict = dict({"Attchd": 0, "BuiltIn": 1, "Basment": 2, "Detchd": 3, "NA": 4, "BuiltIn": 5, "CarPort": 6, "2Types": 7,
                 "Sal": 8, "Sev": 9, "Typ": 10, "Mod": 11, "Min1": 12, "Min2": 13, "Maj1": 14, "Maj2": 15},
                inplace=True)
    data.replace({"Garage Type": Dict}, inplace=True)
    data.replace({"Functional": Dict}, inplace=True)

    """
    Mariela's columns 61-80
    """

    # Fill in missing values
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # Write the file
    data.to_csv('cleaned.csv', index=False)
=======
    data = pd.read_csv('DataPreprocessing/dupAmesHousing.csv') 
    # data = pd.read_csv('dupAmesHousing.csv')
    # Cleaning the Data. Documented in Dataprocessing/datacleaning.md
    data.drop(['PID'], axis=1, inplace=True)
    
    # Write the file
    # data.to_csv('benTest.csv', index=False)

    # last 20 enumerations using dict and replace
    Dict = dict({"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5})
    data.replace({"Garage Qual": Dict, "Garage Cond": Dict}, inplace=True)

    Dict = dict({"Y": 0, "P": 1, "N": 2})
    data.replace({"Paved Drive": Dict}, inplace=True)

    Dict = dict({"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "NA": 4})
    data.replace({"Pool QC": Dict}, inplace=True)

    Dict = dict({"GdPrv": 0, "MnPrv": 1, "GdWo": 2, "MnWw": 3, "NA": 4})
    data.replace({"Fence": Dict}, inplace=True)

    Dict = dict({"Elev": 0, "Gar2": 1, "Othr": 2, "Shed": 3, "TenC": 4, "NA": 5})
    data.replace({"Misc Feature": Dict}, inplace=True)

    Dict = dict({"CWD": 1, "WD ": 0, "VWD": 2, "New": 3, "COD": 4, "Con": 5, "ConLw": 6, "ConLI": 7, "ConLD": 8, "Oth": 9})
    data.replace({"Sale Type": Dict}, inplace=True)

    Dict = dict({"Normal": 0, "Abnorml": 1, "AdjLand": 3, "Alloca": 4, "Family": 5, "Partial": 6})
    data.replace({"Sale Condition": Dict}, inplace=True)

    values = {"Garage Qual": 5, "Garage Cond": 5, "Pool QC": 4, "Fence": 4, "Misc Feature": 5} # to fill NA values
    data.fillna(value=values, inplace=True)

    data.to_csv('benTest.csv', index=False) # write changes to new file
>>>>>>> Stashed changes
