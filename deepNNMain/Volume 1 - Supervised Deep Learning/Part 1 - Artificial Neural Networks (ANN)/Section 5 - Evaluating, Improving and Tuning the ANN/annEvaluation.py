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
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
# print("X_train: ",X_train)
X_test = standardScaler.fit_transform(X_test)
# print("X_test: ",X_test)
###################################################
# preprocessing end
###################################################


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


###################################################
#ANN evaluation (k fold cross validation) start
###################################################
from keras.models import Sequential
from keras.layers import Dense
from kears.layers import Dropout
#from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def buildClassifier():
    #initialize the ANN
    classifier = Sequential()
    #add the first input layer and the first hidden layer
    classifier.add(Dense(units=6, input_dim=11, kernel_initializer="uniform", activation="relu"))
    #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    #dropout regularization to reduce overfitting
    classifier.add(Dropout(p = 0.1))

    #add the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    #dropout regularization to reduce overfitting
    classifier.add(Dropout(p = 0.1))

    #add the output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    #classifier.add(Dense(units=more than 1, kernel_initializer="uniform", activation="softmax"))
    #compile the ANN, definig the way to update the parameters (weight). in this case, stochastic gradient descent
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    
    
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier, batch_size = 10, nb_epoch = 100)
# print("classifier: ", classifier)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("accuracies: ", accuracies)
print("type(accuracies): ", type(accuracies))
#if you want to parallelize the process avobe, modify the code as this URL shows
#https://stackoverflow.com/questions/44982374/cross-val-score-on-windows10-error-with-parallel-computing
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#class RunCross_val_scoreInPrallel():
#    def __init__(self):
#        accuraciesParallel = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#if __name__ == '__main__':
#    RunCross_val_scoreInPrallel()
mean = accuracies.mean()
std = accuracies.std()
###################################################
#ANN evaluation (k fold cross validation) end
###################################################
