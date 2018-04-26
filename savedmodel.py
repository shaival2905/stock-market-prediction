# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:10:47 2018

@author: shaiv
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
    X = pd.DataFrame(X)
    
    # normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled)
    X = X.values
    y = y[27:5010]
    y = pd.DataFrame(y)
  
    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y.values.ravel())
    Y = encoder.transform(y.values.ravel())
    return (X,Y)


# Read the dataset
X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.20, random_state=415)    


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("Testing accuracy:")
scoretr = loaded_model.evaluate(test_x, test_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scoretr[1]*100))
print()
print("Training accuracy:")
scorete = loaded_model.evaluate(train_x, train_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scorete[1]*100))
print()
print("Overall accuracy")
scoreov = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scoreov[1]*100))