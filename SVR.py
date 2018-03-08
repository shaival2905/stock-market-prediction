# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:48:23 2018

@author: shaival

PLEASE READ THE Read_Me.txt file

"""

import math
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import features
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def split(x,y,per):
    i = math.floor(len(x)*per)
    train_x, test_x, train_y, test_y = x[:-i],x[-i:],y[:-i],y[-i:]
    return train_x, test_x, train_y, test_y

def read_dataset():
    X=[]
    y=[]
    simavg, weiavg, momentum, stoK, stoD, rsi, MACD, WR, ado, CCI, monindex, label = features.features("yahoostock.csv")
    df = pd.read_csv("yahoostock.csv")
    
    df = df.iloc[::-1]
    date = df[df.columns[0]]
    close = df[df.columns[5]].values
    close = close[1:]
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
    y.append(close)
    X = np.array(X)
    X = X.transpose()
    date=date.transpose()
    date = date[27:5010]
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
    print(y.shape)

    return (X,y,date)


X,y,date = read_dataset()

train_x, test_x, train_y, test_y = split(X,y,0.10)
train_date, test_date, train_ydate, test_ydate = split(date,y,0.10)
print(len(train_date))
# Fit regression model
svr_lin = SVR(kernel='linear', C=1000)
svr_poly = SVR(kernel='poly', C=1000)
y_lin = svr_lin.fit(train_x, train_y).predict(test_x)
y_poly = svr_poly.fit(train_x, train_y).predict(test_x)
me_poly=0
me_lin=0
for i in range(len(y_poly)):
    tl = test_y[i]-y_lin[i]
    tp = test_y[i]-y_poly[i]
    me_lin = me_lin + abs(tl)
    me_poly = me_poly + abs(tp)

me_poly = me_poly/len(y_poly)
me_lin = me_lin/len(y_lin)
print("Mean error of polynomial: ", me_poly)
print("Mean error of linear: ", me_lin)

plt.scatter(test_date, test_y,  color='black')
plt.plot(test_date, y_lin, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

