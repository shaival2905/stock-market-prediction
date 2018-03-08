# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:13:14 2018

@author: shaival

PLEASE READ THE Read_Me.txt file

"""



import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import features
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense



def split(x,y,per):
    i = math.floor(len(x)*per)
    train_x, test_x, train_y, test_y = x[:-i],x[-i:],y[:-i],y[-i:]
    return train_x, test_x, train_y, test_y

def read_dataset():
    X=[]
    y=[]
    simavg, weiavg, momentum, stoK, stoD, rsi, MACD, WR, ado, CCI, monindex, label = features.features("yahoostock.csv")
    X.append(simavg)
    X.append(weiavg)
    X.append(momentum)
    X.append(stoK)
    X.append(stoD)
    X.append(rsi)
    X.append(MACD)
    X.append(WR)
    X.append(ado)
    X.append(CCI)
    X.append(monindex)
    y.append(label)
    X = np.array(X)
    X = X.transpose()

    X = X[27:5010]
    print(X.shape)
    X = pd.DataFrame(X)
    
    # normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled)
    X = X.values
    y = np.array(y)
    y = y.transpose()
    y = y[27:5010]
    y = pd.DataFrame(y)
    print(y.shape)
    
    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y.values.ravel())
    Y = encoder.transform(y.values.ravel())
    return (X,Y)


# Read the dataset
X, Y = read_dataset()
    
train_x, test_x, train_y, test_y = split(X,Y,0.20)

# create model
model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['mse','accuracy'])
""" 
sgd is Stochastic gradient descent optimizer
it is giving slight better result than adam optimizer and adamax optimizer 

"""
# Fit the model
#print(Y.shape)
model.fit(train_x, train_y, epochs=200, batch_size=10)

# calculate predictions
predict = model.predict(test_x)

# round predictions
round_value = [round(x[0]) for x in predict]
print(round_value)

# evaluate the model
scores = model.evaluate(train_x, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

