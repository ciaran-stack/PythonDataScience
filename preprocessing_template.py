# library import

import pandas as pd
import matplotlib as mplt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv') # turn data.csv into dataframe
x = dataset.iloc[:, :-1].values # get the values of the columns up to not including the final column
y = dataset.iloc[:, -1].values # get the values of the last column

# turn each value into np.nan and find mean for each column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# select columns for which the values of the imputer will be put
imputer.fit(x[:, 1:3])

# send data to x columns
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Turn categorical column data into binary data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # take the first column and set each unqiue category into a binary
X = np.array(ct.fit_transform(x)) # fit and transform the matrix

# Turn the Purchased column (dependent variable) into categorical
le = LabelEncoder()
y = le.fit_transform(y)

# Create training and test data sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Feature Scaling (Avoid features to be dominated)
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)


