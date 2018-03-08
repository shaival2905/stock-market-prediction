# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:42:32 2018

@author: shaival
"""


def calcmonindex(mon,close):
    calcmon=[]
    for i in range(12):
        calcmon.append(0)
        
    for i in range(1,len(close)):
        t = close[i]-close[i-1]
        t = t/close[i-1]
        calcmon[mon[i]-1] = calcmon[mon[i]-1] + t
    
    feature=[]
    feature.append(0)
    for i in range(1,len(close)):
        feature.append(calcmon[mon[i]-1])
        
    return feature

