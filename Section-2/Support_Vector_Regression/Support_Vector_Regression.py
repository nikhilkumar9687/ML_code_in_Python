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
x = dataset.iloc[:, 1:-1].values  # matrix of features #colon (:) means the range ans -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector

print(x)
print(y)

"""
Converting Y(Salary) to a 2d array as fit.transform 
excepts a 2d array for feature scaling
"""

y = y.reshape(len(y),1)
print(y)

"""
Feature Scaling
"""
from sklearn.preprocessing import StandardScaler

stndrd_sclr_x = StandardScaler()
stndrd_sclr_y = StandardScaler()
x = stndrd_sclr_x.fit_transform(x)
y = stndrd_sclr_y.fit_transform(y)

print(x)
print(y)

"""
Training The SVR model
"""
from sklearn.svm import SVR

sv_regressor = SVR(kernel = 'rbf')
sv_regressor.fit(x, y)

"""
Predict The New Results
"""

y_transfrom = stndrd_sclr_y.inverse_transform( sv_regressor.predict(stndrd_sclr_x.transform([[6.5]])))
print(y_transfrom)

"""
Visualising the SVR results
"""
plt.scatter(stndrd_sclr_x.inverse_transform(x),
            stndrd_sclr_y.inverse_transform(y),
            color = 'red')
plt.plot(stndrd_sclr_x.inverse_transform(x),
         stndrd_sclr_y.inverse_transform(sv_regressor.predict(x)),
         color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
Visualising the SVR results for Higher Resolution
and Smoother curve 
"""

x_grid = np.arange(min(stndrd_sclr_x.inverse_transform(x)),
                   max(stndrd_sclr_x.inverse_transform(x)),
                   0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(stndrd_sclr_x.inverse_transform(x),
            stndrd_sclr_y.inverse_transform(y),
            color = 'red')

plt.plot(x_grid,
         stndrd_sclr_y.inverse_transform(sv_regressor.predict(stndrd_sclr_x.transform(x_grid))),
         color = 'blue')

plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
