"""
main.py
Run this file to generate all the graphs needed for the project
"""

from DataPreprocessing.dataCleaning import data_cleaning
from RFR import rfr
from RFR_RFE import rfr_rfe
from SVR import svr
from SVR_RFE import svr_rfe
from SVR_RFE_Manual import svr_rfe_manual

if __name__ == "__main__":
    print("Cleaning data...")
    data_cleaning()
    print("Cleaned data saved as cleaned.csv")

    print("\nCreating graph for Random Forest Regression with no feature selection...")
    print("(this takes a few seconds)")
    rfr()
    print("Graph done and saved as RFR_BASE.png in Graphs folder!")

    print("\nCreating graph for Support Vector Regression with no feature selection...")
    print("(this takes a few seconds)")
    svr()
    print("Graph saved as SVR_BASE.png in Graphs folder!")

    print("\nCreating graph for Random Forest Regression with Recursive Feature Elimination...")
    print("(this takes a minute or two)")
    rfr_rfe()
    print("Graph saved as RFR_RFE.png in Graphs folder!")

    print("\nCreating graph for Support Vector Regression with Recursive Feature Elimination...")
    print("(this takes a minute or two)")
    svr_rfe()
    print("Graph saved as SVR_RFE.png in Graphs folder!")

    print("\nCreating graph for Support Vector Regression with Recursive Feature Elimination except the RFE is partial built ins and implemented manually...")
    print("(this takes quite a bit of time)")
    svr_rfe_manual()
    print("Graph saved as SVR_RFE_M.png in Graphs folder!")
