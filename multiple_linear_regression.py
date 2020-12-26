# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data import and columns
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values # set x-input columns
y = dataset.iloc[:, -1].values # set y-output

# Encode the categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # take the fourth column and set each unqiue category into a binary
X = np.array(ct.fit_transform(x)) # fit and transform the matrix

# Dataset split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# training multiple reg model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2) # used for precision (number of decimal places)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), axis=1))

# **************************************** Model Evaluation *******************************************************
score = r2_score(y_test, y_pred)
print(score)