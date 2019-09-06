# -*- coding: utf-8 -*-
#recurrent NN

############################################
#Step 1 Data preprocessing
############################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the training dataset
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
#print("training_set: ",training_set)
#print("type(training_set): ",type(training_set))
#print("dataset_train.iloc[:, 1].values: ", dataset_train.iloc[:, 1].values)
#print("type(dataset_train.iloc[:, 1].values): ", type(dataset_train.iloc[:, 1].values))
#print("type(dataset_train.iloc[:, 1:2]): ", type(dataset_train.iloc[:, 1:2]))

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
#fit = get the min and max value of the argument
#transform = calc the normalized values
training_set_scaled = sc.fit_transform(training_set)

#create a data structure with 60 timesteps and 1 output
X_train , y_train = [], []
for i in range (60, training_set_scaled.shape[0]):
#    one array data
    X_train.append(training_set_scaled[(i-60):i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train , y_train = np.array(X_train), np.array(y_train)

#reshaping
#see https://keras.io/layers/recurrent/#lstm for the meaning of the second argument
#3D tensor with shape (batch_size, timesteps, input_dim).
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#print(X_train.shape)

############################################
#Step 1 Data preprocessing end
############################################

############################################
#Step 2 Building the RNN
############################################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the RNN
regressor = Sequential()

#add the first LSTM layer
#since we add another LSTM layer, return_sequences=True
#units = num of dimensions (layers)
regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
#add dropout regularization
regressor.add(Dropout(0.2))

#add the second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
#dropout regularization
regressor.add(Dropout(0.2))

#add the third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
#dropout regularization
regressor.add(Dropout(0.2))

#add the fourth LSTM layer
regressor.add(LSTM(units=50, return_sequences=False))
#dropout regularization
regressor.add(Dropout(0.2))

#add the output layer
regressor.add(Dense(units=1))

#compiling (defining the way of update/back propagation) the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fit the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size=32)
############################################
#Step 2 Building the RNN end
############################################

############################################
#Step 3 Making the prediction and visualizing the results
############################################

#get the real stock price in 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

#get the predicted stock price in 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#print(dataset_total)

#extract the necessary input data for the prediction of dataset_test
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#print(inputs)
inputs = inputs.reshape(-1,1)
#print(inputs)
#scale the input value. You must not use fit_transform (https://mathwords.net/fittransform)
inputs = sc.transform(inputs)
#print(inputs)

#create a data structure with 60 timesteps and 1 output for testing
X_test = []
for i in range (60, 60 + real_stock_price.shape[0]):
#    one array data
    X_test.append(inputs[(i-60):i, 0])
X_test = np.array(X_test)
#print(X_test.shape)
#reshaping
#3D tensor with shape (batch_size, timesteps, input_dim).
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#run prediction
predicted_stock_price = regressor.predict(X_test)
#convert the scaled values to the original value
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualize the result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time (Day)')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

#evaluate the prediction compared to the real stock price
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print("rmse: ", rmse)

#divide rmse with predicted_stock_price
#relativeRmse = 
overall_error_rate = rmse / np.mean(predicted_stock_price)
print('overall_error_rate: ',overall_error_rate)
