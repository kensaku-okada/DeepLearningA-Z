# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
#x = dataset.iloc[:, 0:3].values
x = dataset.iloc[:, :-1].values
#print(x)
y = dataset.iloc[:, -1].values


#fill the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#print(imputer)

#choose only the columns having missing data
imputerNanFilled = imputer.fit(x[:, [1,2]])
#print(imputerNanFilled)
x[:, [1,2]] = imputerNanFilled.transform(x[:, [1,2]])
#print(x[:, [1,2]])


#encode categorical data into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#convert the categorical data
labelEncoder_x = LabelEncoder() 
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
#print(oneHotEncoder.fit_transform(x).toarray())
x = oneHotEncoder.fit_transform(x).toarray()

#convert the output (dependent variable)
labelEncode_y =LabelEncoder()
y = labelEncode_y.fit_transform(y)


#split the dataset into the train and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 


#feature scaling
from sklearn.preprocessing import StandardScaler
#x_train and x_test should be scaled on the same basis
standardScalerX = StandardScaler()
x_train_scaled = standardScalerX.fit_transform(x_train)
x_test_scaled = standardScalerX.transform(x_test)
#y data does not have to be scales because it contains only classification binary data







