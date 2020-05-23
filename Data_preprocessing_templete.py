"""
Importing Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Importing Dataset
"""
dataset = pd.read_csv('Data.csv') # the name of dataset might change every time
x = dataset.iloc[:, :-1].values  # matrix of features #colon (:) means the range ans -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector


"""
Splitting the data into Training set and Test set
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

