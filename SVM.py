# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:48:23 2018

@author: shaival

PLEASE READ THE Read_Me.txt file

"""
import math
from sklearn import svm
import features
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def read_dataset():
    
    X,y = features.features("yahoostock.csv")
    
    X = X[27:5010]
    print(X.shape)
    
    X = pd.DataFrame(X)
   # print(X)
    #X.to_csv('data.csv')
    #min_max_scaler = preprocessing.MinMaxScaler()
    #np_scaled = min_max_scaler.fit_transform(X)
    #X = pd.DataFrame(np_scaled)
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

print(pd.DataFrame(train_x))
clf = svm.SVC(kernel='poly', degree=2, gamma=0.01, C=1000)
clf.fit(train_x,train_y)    

pred=clf.predict(test_x)
accuracy = accuracy_score(pred,test_y)
print("Test Accuracy: ",accuracy)


  