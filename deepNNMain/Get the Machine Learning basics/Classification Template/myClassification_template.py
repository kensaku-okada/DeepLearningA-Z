# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
#x = dataset.iloc[:, 0:3].values
x = dataset.iloc[:, [2,3]].values
#print(x)
y = dataset.iloc[:, -1].values


#fill the missing data -> no missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
##choose only the columns having missing data
#imputerNanFilled = imputer.fit(x[:, [1,2]])
#x[:, [1,2]] = imputerNanFilled.transform(x[:, [1,2]])

##encode categorical data into numbers -> no categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##convert the categorical data
#labelEncoder_x = LabelEncoder() 
#x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
#oneHotEncoder = OneHotEncoder(categorical_features=[0])
##print(oneHotEncoder.fit_transform(x).toarray())
#x = oneHotEncoder.fit_transform(x).toarray()
##convert the output (dependent variable)
#labelEncode_y =LabelEncoder()
#y = labelEncode_y.fit_transform(y)


#split the dataset into the train and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) 

#feature scaling
from sklearn.preprocessing import StandardScaler
#x_train and x_test should be scaled on the same basis
standardScalerX = StandardScaler()
x_train_scaled = standardScalerX.fit_transform(x_train)
x_test_scaled = standardScalerX.transform(x_test)
#y data does not have to be scales because it contains only classification binary data


#fit logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train_scaled, y_train)


#predict the test set results
y_pred = classifier.predict(x_test_scaled)


#make a confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_pred)


#visualize the training set result
from matplotlib.colors import ListedColormap
x_set, y_set = x_train_scaled, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('logistic regression (training set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()


#visualize the test set result
x_set, y_set = x_test_scaled, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('logistic regression (test set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()


