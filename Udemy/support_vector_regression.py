import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#  import data set
dataset = pd.read_csv('Position_Salaries.csv') # data frame
x = dataset.iloc[:, 1:-1].values # x_values
y = dataset.iloc[:, -1].values

# ************************************* Feature Scaling *************************************
y = y.reshape(len(y), 1) # transform y to an array of column values (2d array) with 1 column
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(x)
y = sc_Y.fit_transform(y)


# ************************************* Training SVR Model *************************************
regressor = SVR(kernel='rbf') # creates SVR model with radial basis function
regressor.fit(X, y)

# ************************************* Predicting a new result *************************************
regressor.predict(sc_X.transform([[6.5]])) # we scale 6.5 with sc_X because all x features are scaled to low range
m = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))) # reverse scaling of y to get original (inverse transform method)

# ************************************* Visualization *************************************
# plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red')
# plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue')
# plt.title('SVR Position Salaries')
# plt.xlabel('Position Label')
# plt.ylabel('Salary')
# plt.show()

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.transform(X)), .01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue')

plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

