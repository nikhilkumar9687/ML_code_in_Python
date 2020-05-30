"""
Decision Tree Regression
"""
"""
Importing The Libraries
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing Dataset
"""

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""
Training The Decision Tree Regression model on whole Dataset
"""

from sklearn.tree import DecisionTreeRegressor

d_t_reg = DecisionTreeRegressor(random_state= 0)
d_t_reg.fit(x, y)

"""
Predicting the Results
"""

print(d_t_reg.predict([[6.5]]))

"""
Visualising the Decision Tree Regression (higher Resolution)
"""

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, d_t_reg.predict(x_grid), color = "blue")
plt.title('Truth Or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
Visualising the Decision Tree Regression (low Resolution)
"""

plt.scatter(x, y, color = 'red')
plt.plot(x, d_t_reg.predict(x), color = "blue")
plt.title('Truth Or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

