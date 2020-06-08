# K Nearest Neighbors

"""
Importing Libraries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Importing Dataset
"""
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, : -1].values
y = dataset.iloc[ :, -1].values

"""
Splitting the dataset on Training Set and Test Set
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size= 0.25,
                                                    random_state=0)
"""
Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""
Training the K_NN Model on Training set
"""

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors= 5,
                                      p=2,
                                      metric='minkowski')

knn_classifier.fit(x_train, y_train)

"""
Predicting the New Result
"""

y_pred = knn_classifier.predict(sc.transform([[30,87000]]))
print (y_pred)
"""
Predicting The test Result
"""
y_pred = knn_classifier.predict(x_test)
#np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                      y_test.reshape(len(y_test), 1)),
                    1))

"""
Making the Confusion Matrix
"""
from sklearn.metrics import confusion_matrix, accuracy_score
con_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")
print(con_mat)
acc_src = accuracy_score(y_test, y_pred)
print(acc_src)


"""
Visualising the Training set results
"""

from matplotlib.colors import ListedColormap
x_set, y_set = sc.inverse_transform(x_train), y_train

x1, x2 =np.meshgrid(np.arange(start= x_set[:, 0].min() - 10,
                              stop= x_set[:, 0].max() + 10,
                              step = 0.25),
                    np.arange(start= x_set[:, 1].min() - 1000,
                              stop=  x_set[:, 1].max() + 1000,
                              step=  0.25))
plt.contourf(x1, x2, knn_classifier.predict(sc.transform(np.array([x1.ravel(),
                                                                            x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red','white'))(i),
                label = j)

plt.title('K_NN (Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

"""
Visualising the Test set results
"""

from matplotlib.colors import ListedColormap
x_set, y_set = sc.inverse_transform(x_test), y_test

x1, x2 =np.meshgrid(np.arange(start= x_set[:, 0].min() - 10,
                              stop= x_set[:, 0].max() + 10,
                              step = 0.25),
                    np.arange(start= x_set[:, 1].min() - 1000,
                              stop=  x_set[:, 1].max() + 1000,
                              step=  0.25))
plt.contourf(x1, x2, knn_classifier.predict(sc.transform(np.array([x1.ravel(),
                                                                            x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red','white'))(i),
                label = j)

plt.title('K_NN (Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


