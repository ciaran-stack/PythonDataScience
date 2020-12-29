# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# read data set
dataset = pd.read_csv('Salary_Data.csv') # set data frame
x = dataset.iloc[:, :-1].values # set x-input columns
y = dataset.iloc[:, -1].values # set y-output

# Split Dataset into two parts
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=0)

# Train model on data set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Test model on data set
y_pred = regressor.predict(x_test)

# visualize result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()