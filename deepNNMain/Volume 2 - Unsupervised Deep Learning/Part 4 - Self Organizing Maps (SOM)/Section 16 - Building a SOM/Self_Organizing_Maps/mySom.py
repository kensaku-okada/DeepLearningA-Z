

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#1print(os.getcwd())

#"Class" attribute: 1 means approved, 0 means not approved
dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
#print(type(X))
y = dataset.iloc[:, -1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
#normalized X
X = sc.fit_transform(X)

#train the SOM with a third party library (https://pypi.org/project/MiniSom/#history), which is minisom.py
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len= 15, sigma = 1.0, learning_rate=0.5)

#initialize the weights
som.random_weights_init(X)

#train the som
som.train_random(data = X, num_iteration=100)

#visualize the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
#som.distance_map returns all MID (mean into winning node distances) for all nodes. The higher the more likely to be fraud (or outlier).
pcolor(som.distance_map().T)
colorbar()
#add markers
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate (X):
    w = som.winner(x)
#    if the customer gets approval, y[i] = 1 and markers[y[i]] = 's', if the customer did not get approval, y[i] = 0 and markers[y[i]] = 'o'
    plot(w[0] + 0.5, w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#find the frauds
mappings = som.win_map(X)
#You need to manually determine the outlier winning node coordinate from the visualized figure
#potentialFrauds = np.concatenate((mappings[7,4]), axis = 0)
potentialFrauds = mappings[7,4]
#unscale the elements of all customers
unscaledPotentialFrauds = sc.inverse_transform(potentialFrauds)









