# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
###################################################
# preprocessing start
###################################################
X = dataset.iloc[:, 3:13].values
# print("X: ", X)
y = dataset.iloc[:, 13].values
# print("y: ", y)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#reference: https://qiita.com/yoshimo123/items/3717bd17ba74764dbc69
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# print("X: ",X)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# print("X: ",X)

#convert the categorial data into the binary dataset
oneHotEncoder = OneHotEncoder(categorical_features=[1])
#X1 = oneHotEncoder.fit_transform(X)
#print("X1: ",X1)
X = oneHotEncoder.fit_transform(X).toarray()
# print("X: ",X)

#avoid dummy variable trap (remove one column from the categorical dummy variables)
X = X[:, 1:]


#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

#divide the data into the training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X_train.shape: ",X_train.shape)
print("X_test.shape: ",X_test.shape)
print("y_test.shape: ",y_test.shape)

#feature scaling (standardadize the input variables. see http://ailaby.com/scaler/)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# print("X_train: ",X_train)
X_test = sc.transform(X_test)
#X_test = sc.fit_transform(X_test)
# print("X_test: ",X_test)
###################################################
# preprocessing end
###################################################

###################################################
#ANN all hyperparameters tuning (num of epohs, batch size, num of layers, k-fold cross validation) start
###################################################
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def buildClassifier(optimizer):
    #initialize the ANN
    classifier = Sequential()
    #add the first input layer and the first hidden layer. the num of nodes at this layer = units
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    #add the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

    #add the output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    #classifier.add(Dense(units=more than 1, kernel_initializer="uniform", activation="softmax"))
    #compile the ANN, definig the way to update the parameters (weight). in this case, stochastic gradient descent
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier)
# print("classifier: ", classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

#fit with the training data
grid_search = grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("best_params: ", best_params)
best_accuracy = grid_search.best_score_
print("best_accuracy: ", best_accuracy)
