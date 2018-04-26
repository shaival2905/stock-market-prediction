# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:13:14 2018

@author: shaival

PLEASE READ THE Read_Me.txt file

"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import features
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

def read_dataset():
    
    X,y = features.features("yahoostock.csv")
    X = X[27:5010]
    print(X.shape)
    X = pd.DataFrame(X)
    
    # normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled)
    X = X.values
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
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.20, random_state=415)    

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
print("training accuracy")
scores = model.evaluate(train_x, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Model Saved")
 