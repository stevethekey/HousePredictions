import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

if __name__ == "__main__":
    # set data = the csv file
    data = pd.read_csv('AmesHousing.csv')

    # Cleaning the Data. Documented in Dataprocessing/datacleaning.md

    """
    Steven's 1-20th columns
    """
    data.drop(['PID', 'Order'], axis=1, inplace=True)

    ms_zoning_mapper = {'A': 0, 'A (agr)': 0, 'C': 1, 'C (all)': 1, 'FV': 2, 'I': 3, 'I (all)': 3, 'RH': 4, 'RL': 5, 'RP': 6, 'RM': 7}
    data['MS Zoning'].replace(ms_zoning_mapper, inplace=True)

    driveway_type_mapper = {'Grvl': 0, 'Pave': 1}
    data['Street'].replace(driveway_type_mapper, inplace=True)

    alley_type_mapper = {'Grvl': 0, 'Pave': 1, 'NA': 2}
    data['Alley'].replace(alley_type_mapper, inplace=True)

    lot_shape_mapper = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
    data['Lot Shape'].replace(lot_shape_mapper, inplace=True)

    land_contour_mapper = { 'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}
    data['Land Contour'].replace(land_contour_mapper, inplace= True)

    utilities_mapper = {'AllPub': 0, 'NoSewr': 1, 'NoSeWa': 2, 'ELO': 3}
    data['Utilities'].replace(utilities_mapper, inplace=True)

    lot_config_mapper = {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3,  'FR3': 4 }
    data['Lot Config'].replace(lot_config_mapper, inplace= True)

    land_slope_mapper = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    data['Land Slope'].replace(land_slope_mapper, inplace= True)

    neighborhood_mapper = {'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4, 'CollgCr': 5,
    'Crawfor': 6, 'Edwards': 7, 'Gilbert': 8, 'IDOTRR': 9, 'MeadowV': 10, 'Mitchel': 11, 'Names': 12, 'NoRidge': 13,
    'NPkVill': 14, 'NridgHt': 15, 'NWAmes': 16, 'NAmes': 16, 'OldTown': 17, 'SWISU': 18, 'Sawyer': 19, 'SawyerW': 20, 'Somerst': 21,
    'StoneBr': 22, 'Timber': 23, 'Veenker':24, 'Greens': 25, 'GrnHill': 26, 'Landmrk': 27}
    data['Neighborhood'].replace(neighborhood_mapper, inplace= True)

    first_condition_mapper = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA':6, 'RRNe': 7, 'RRAe': 8}
    data['Condition 1'].replace(first_condition_mapper, inplace=True)

    second_condition_mapper = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA':6, 'RRNe': 7, 'RRAe': 8}
    data['Condition 2'].replace(second_condition_mapper, inplace=True)


    building_type_mapper = {'1Fam': 0, '2FmCon': 1, '2fmCon': 1, 'Duplx': 3, 'Duplex': 3, 'TwnhsE': 4, 'TwnhsI': 5, 'Twnhs': 6}
    data['Bldg Type'].replace(building_type_mapper, inplace= True)

    house_style_mapper = {'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7}
    data['House Style'].replace(house_style_mapper, inplace= True)
 

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

    garage_finish_mapper = {'Fin': 0, 'RFn': 1, 'Unf': 2, 'NA': 3}
    data['Garage Finish'].replace(garage_finish_mapper, inplace=True)

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

    # Fill in missing values
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # Remove Correlated Features
    correlated_features = set()
    correlation_matrix = data.drop('SalePrice', axis=1).corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    print(correlated_features)
    for feature in correlated_features:
        data.drop(feature, axis=1, inplace=True)

    # Write the file
    data.to_csv('cleaned.csv', index=False)

    """
    code to check if every attribute is a string. not finished

    data_clean = pd.read_csv('cleaned.csv')

    for column in data_clean.columns:
        for attribute in data_clean[column]:
            print("the attribute " + attribute + "'s type is " + type(attribute))

            if(type(attribute) != int)
                this means a problem is here

    - steve
    """
