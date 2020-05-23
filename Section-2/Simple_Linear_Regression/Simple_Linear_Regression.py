"""
Importing Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing Dataset
"""
dataset = pd.read_csv('Salary_Data.csv')  # the name of dataset might change every time
x = dataset.iloc[:, :-1].values  # matrix of features #colon (:) means the range ans -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector

"""
Splitting the data into Training set and Test set
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

"""
Training the Simple Linear Regression Model on the Training set
"""

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(x_train, y_train)

"""
Predicting the Test Results
"""

y_pridct = linear_regression.predict(x_test)

"""
Visualising the Training set results
"""

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, linear_regression.predict(x_train), color='blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
#plt.interactive(False)
"""
Visualising the Test set results
"""

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, linear_regression.predict(x_train), color='blue')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
#plt.interactive(False)

"""
Making a single prediction (for example salary of a 12 years experience person)
using [[]] because it is a 2d array representation as needed by predict method

12 --> scalar
[12] --> 1D array
[[12]] --> 2D array

"""

print(linear_regression.predict([[12]]))

"""
Getting the Final Liner regression equation with values of coefficients
"""

print( linear_regression.coef_)
print(linear_regression.intercept_)

