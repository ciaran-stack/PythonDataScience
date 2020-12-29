import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Decision Tree are not great for single feature datasets

# import data sets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training the Decision Tree Regression Model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y) # trains decision tree regressor on X and y

# ***************************** prediction ***********************************
regressor.predict([[6.5]]) # no feature scaling needed

# ************************** Visualization ***********************************
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Position Salary Decision Tree Vis')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



