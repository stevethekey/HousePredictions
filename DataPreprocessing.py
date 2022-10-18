import pandas as pd

ames_housing = pd.read_csv("AmesHousing.csv", na_values='?')
ames_housing = ames_housing.drop(columns="PID")

ames_housing.head()
target_name="SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]


#data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]


#https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html

