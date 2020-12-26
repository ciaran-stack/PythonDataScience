import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
# Data import and column selection
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# training set declaration
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

# feature scaling for age and salary
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# train and test using sklearn API & documentation
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# test prediction line
# print(classifier.predict(sc.transform([[30, 87000]])))

# *************************** predicting test set results ************************************************
y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), axis=1))

cm = confusion_matrix(y_test, y_pred) # confusion matrix
a_score = accuracy_score(y_test, y_pred) # accuracy of predictions

# ******************************** Visualization ********************************

# Visualize Training Set
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=.25),
                     np.arange(start=X_set[:,1].min()-1000, stop=X_set[:,1].max()+1000, step=.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j, 0], X_set[y_set==j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression Training Set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()



# Visualize Test Set