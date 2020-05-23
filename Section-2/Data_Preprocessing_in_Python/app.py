"""
Importing Libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing Dataset
"""
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  # matrix of features #colon (:) means the range ans -1 is the last column
y = dataset.iloc[:, -1].values  # y is dependent variable vector

print(x)
print(y)

"""
Taking care of missing data if any in the dataset
"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)
"""
Encoding Independent data 
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

col_trnsfrm = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(col_trnsfrm.fit_transform(x))

print(x)
"""
Encoding Dependent data 
"""

from sklearn.preprocessing import LabelEncoder

lb_encoder = LabelEncoder()
y = lb_encoder.fit_transform(y)

print(y)

"""
Splitting the data into Training set and Test set
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

"""
feature scaling
"""

from sklearn.preprocessing import StandardScaler

stndrd_sclr = StandardScaler()
x_train[:, 3:] = stndrd_sclr.fit_transform(x_train[:, 3:])
x_test[:, 3:] = stndrd_sclr.transform(x_test[:, 3:])

print(x_train)
print(x_test)

