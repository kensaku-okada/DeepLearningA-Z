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
print("X: ", X)
y = dataset.iloc[:, 13].values
print("y: ", y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#reference: https://qiita.com/yoshimo123/items/3717bd17ba74764dbc69
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
print("X: ",X)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print("X: ",X)

#convert the categorial data into the binary dataset
oneHotEncoder = OneHotEncoder(categorical_features=[1])
#X1 = oneHotEncoder.fit_transform(X)
#print("X1: ",X1)
X = oneHotEncoder.fit_transform(X).toarray()
print("X: ",X)

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
print("X_train: ",X_train)
X_test = standardScaler.fit_transform(X_test)
print("X_test: ",X_test)

###################################################
# preprocessing end
###################################################

###################################################
#making artifitial neural network (ANN) start
###################################################
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialize the ANN
classifier = Sequential()

#add the first input layer and the first hidden layer
#the following code making a warnming message because Keras has been updated to 2.0 after the lectures were created. The code will still work since Keras is backward compatible.
#To fix this warning, do as the message says:
#classifier.add(Dense(units=6, input_dim=11, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#add the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

#add the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
#classifier.add(Dense(units=more than 1, kernel_initializer="uniform", activation="softmax"))

#compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit the ANN to the tarining data set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
###################################################
#making artifitial neural network (ANN) end
###################################################

###################################################
#making the prediction start
###################################################
#predict the test set result
y_pred = classifier.predict(X_test)

#cinverge the probability 0 to 1 to boolean at the threshold
threshold = 0.5
y_pred_bool = (y_pred > threshold)
###################################################
#making the prediction end
###################################################

#make the confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrixResult = confusion_matrix(y_test, y_pred_bool)
#print(confusionMatrixResult)


###################################################
#Homework solution: https://www.udemy.com/deeplearning/learn/v4/t/lecture/6743686?start=0

#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
###################################################
#compare the variable dataset with X to figure out the way to convert categorical data
X_real1 = [[0., 0., 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
#scale the data by 0 mean and 1 sandard deviation
standardScaler = StandardScaler()
X_real1_scaled = standardScaler.fit_transform(X_real1)
y_real1_scaled_pred = classifier.predict(X_real1_scaled)
print("y_real1_scaled_pred: ",y_real1_scaled_pred)
y_real1_scaled_pred_bool = (y_real1_scaled_pred > 0.5)
