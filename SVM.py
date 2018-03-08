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
from sklearn.model_selection import cross_val_score

def split(x,y,per):
    i = math.floor(len(x)*per)
    train_x, test_x, train_y, test_y = x[:-i],x[-i:],y[:-i],y[-i:]
    return train_x, test_x, train_y, test_y

def read_dataset():
    X=[]
    y=[]
    simavg, weiavg, momentum, stoK, stoD, rsi, MACD, WR, ado, CCI,monindex, label = features.features("yahoostock.csv")
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

print(train_x)
clf = svm.SVC(gamma=0.01, C=1000)
clf.fit(train_x,train_y)    

pred=clf.predict(test_x)
accuracy = accuracy_score(pred,test_y)
print("Test Accuracy: ",accuracy)


  