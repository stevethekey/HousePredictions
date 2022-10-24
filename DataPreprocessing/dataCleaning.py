import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # set data = the csv file
    data = pd.read_csv('AmesHousing.csv')

    # Cleaning the Data. Documented in Dataprocessing/datacleaning.md
    data.drop(['PID', 'Order'], axis=1, inplace=True)

    """
    Ben's columns
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
    Gabe's columns
    """

    # Fill in missing values
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # Write the file
    data.to_csv('cleaned.csv', index=False)
