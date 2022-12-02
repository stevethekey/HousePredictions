# HousePredictions
## Project By: William Steven Matiz, Gabriel Mourad, Mariela Badillo, Benjamin Zech
Our project for Intro to Data Mining - CIS 4930 at Florida State University

### Running the project
    Before running, make sure you have the required libraries.
    To do this, run the command: 
    `pip install -r requirement.txt`

    To run the project:
    `python main.py`

### Files
**AmesHousing.csv**: Data set for the housing

**data_description.txt**: Describes what the values in AmesHousing's features mean

**DataPreprocessing**  
|  
|---- **dataCleaning.py**: Code that cleans the AmesHousing.csv  
|  
|---- **datacleaning.md**: Markdown file describing changes made for the data cleaning process  
\
**RFR-RFE-Manual.py**: Random Forest Regressor(RFR) with Recursive Feature Elimination(RFE). This version of RFE is implemented partially with built-ins and partially manually

**RFR-RFE.py**: Random Forest Regressor(RFR) with Recursive Feature Elimination(RFE) implemented using built-ins

**RFR.py**: Random Forest Regressor(RFR) with no feature selection

**SVR-RFE-Manual**: Support Vector Regression(SVR) with recursive feature elimination(RFE). This version of RFE is implemented partially with built-ins and partially manually

**SVR-RFE.py**: Support Vector Regression(SVR) with Recursive Feature Elimination(RFE) implemented using built-ins

**SVR.py**: Support Vector Regression(SVR) with no feature selection

**UNF.py**: Univariance Feature Selection(UNF), UNF with RFR, and UNF with SVC all implented using built-ins

**main.py**: Main file to generate all results and graphs of the project

### Libraries and Modules
In addition to the standard python library, numpy, pandas, matplotlib, sklearn were used

### Sources
**Code Citation**:
    - https://towardsdatascience.com/application-of-feature-selection-techniques-in-a-regression-problem-4278e2efd503

**Sklearn documentation**:
    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

**Data Cleaning**:
    - https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15#:~:text=Feature%20Selection%20in%20Python%20%E2%80%94%20Recursive%20Feature%20Elimination,can%20finally%20begin.%20...%204%204.%20Conclusion%20

**RFE**:
    - https://machinelearningmastery.com/rfe-feature-selection-in-python/
    - https://www.kaggle.com/code/carlmcbrideellis/recursive-feature-elimination-rfe-example/notebook
    - https://www.blog.trainindata.com/recursive-feature-elimination-with-python/
    - https://towardsdatascience.com/powerful-feature-selection-with-recursive-feature-elimination-rfe-of-sklearn-23efb2cdb54e

