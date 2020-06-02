"""
Random Forest Regression
"""

"""
Importing Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing Dataset
"""

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""
Training the Random Forest Regression
"""
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators= 10, random_state= 0)
rf_reg.fit(x, y)

"""
Predicting the New Result
"""
print(rf_reg.predict([[6.5]]))

"""
Visualizing The Random Forest Regression Results (Higher Resolution)
"""

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, rf_reg.predict(x_grid), color = "blue")
plt.title('Truth vs Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
