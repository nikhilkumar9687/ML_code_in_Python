"""
Importing Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Importing Dataset
"""
dataset = pd.read_csv('50_Startups.csv')  # the name of dataset might change every time
x = dataset.iloc[:, :-1].values  # matrix of features #colon (:) means the range ans -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector
print(x)
"""
Encoding categorical data
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

"""
Splitting the data into Training set and Test set
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

"""
Training the Multiple Regression Model on the training set
"""

from sklearn.linear_model import LinearRegression

mul_linr_reg = LinearRegression()
mul_linr_reg.fit(x_train, y_train)

"""
Predicting the Test Results
"""

y_pred = mul_linr_reg.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate(
    (y_pred.reshape(len(y_pred), 1),
     y_test.reshape(len(y_test), 1)),
    axis=1)
)
