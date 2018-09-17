# -*- coding: utf-8 -*-

#hybrid deep learning model

#part 1 identify teh faruds with the SOM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#find the frauds
mappings = som.win_map(X)
#TODO: you need to manuall modify the coordinates of mappings[(4,3)], mappings[(8,4)] by looking at the som visual figure
potentialFrauds = np.concatenate((mappings[(4,3)], mappings[(8,4)]), axis = 0)
#unscale the elements of all customers
unscaledPotentialFrauds = sc.inverse_transform(potentialFrauds)

####################################################
#part 2 go from unsupervised to supervised learning
####################################################

#create the matrix of features (indepependent variable)
#the customer ID wont help detecting faruds, but the approval descision may help
customers = dataset.iloc[:, :-1].values

#create the dependent variables
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0 ] in unscaledPotentialFrauds:
        is_fraud[i] = 1


#train the ANN
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

#this layer was omitted just for simplicity 
## Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#the numbers of dataset (690) and the features (15) are small for ANN, we do not need so many epochs 
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predict the Test set results
y_pred = classifier.predict(customers)
#y_pred is originally a two dimentional array
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
#sort by the probability
y_pred_sorted = y_pred[y_pred[:, 1].argsort()]




