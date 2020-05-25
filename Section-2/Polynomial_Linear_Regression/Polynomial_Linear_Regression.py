"""
Importing Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing Dataset
"""
dataset = pd.read_csv('Position_Salaries.csv')  # the name of dataset might change every time
x = dataset.iloc[:, 1:-1].values  # matrix of features #colon (:) means the range and -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector

"""
Training The linear regression model on dataset
"""
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x, y)

"""
Training The Polynomial regression model on dataset
"""
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

"""
Visualising The Linear regression Results
"""

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
Visualising The Polynomial regression Results
"""
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
Visualising The Polynomial regression Results (with More precession and smoother Curve)
"""
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
Predicting a new Result with Linear Regression
"""
lin_pred = lin_reg.predict([[6.5]])
print(lin_pred)

"""
Predicting a new Result with Polynomial Regression
"""
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_pred)

