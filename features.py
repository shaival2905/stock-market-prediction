# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:15:48 2018

@author: shaiv
"""

import numpy as np
import pandas as pd
import calcmonindex

def maxa(x):
    maxa=0
    for i in range(len(x)):
        if(x[i]>maxa):
            maxa=x[i]
            
    return maxa

def mina(x):
    mina=x[0]
    for i in range(len(x)):
        if(x[i]<mina):
            mina=x[i]
            
    return mina

def avg(d):
    ans=0
    for i in range(len(d)):
        ans = ans + d[i];
    ans = ans/(len(d))
    return ans

def weigavg(d):
    ans1=0
    ans2=0
    n=1
    for i in range(len(d)):
       ans1 = ans1 + n*d[i]
       ans2 = ans2 + n
       n=n+1
       
    ans = ans1/ans2
    return ans
def features(dataset):
    df = pd.read_csv(dataset)
    #date = df[df.columns[0]]
    #df = df.iloc[::-1]
    date=pd.to_datetime(df['Date'])
    
    mon = date.dt.month
    mon = mon.values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    monindex = calcmonindex.calcmonindex(mon,close) #It shows the trend in particular month
    monindex = np.array(monindex)
    Hh=[]
    Ll=[]
    # stockastic oscillators K% and D%
    stoK=[] 
    stoD=[]
    # here moving average is taken for 10 days
    simavg=[] #simple moving average
    weiavg=[] #weighted moving average
    Up=[]
    Dn=[]
    rsi=[] #relative strength index
    ado=[] #Accumulation/Distribution oscillator
    Mi=[]
    SMi=[]
    Di=[]
    CCI=[] #Commodity channel index
    WR=[] #Larry William’s R%
    EMA12=[] #exponential moving average of 12 days
    EMA26=[] #exponential moving average of 26 days
    DIFF=[]
    MACD=[] #Moving average convergence divergence 
    label=[] #It is trend for the next day 1: up and -1: down
    ROC=[] #Rate of change
    
    for i in range(9):
        Hh.append(0)
        Ll.append(0)
        stoK.append(0)
        stoD.append(0)
        simavg.append(0)
        weiavg.append(0)
        rsi.append(0)
        SMi.append(0)
        Di.append(0)
        CCI.append(0)
        WR.append(0)
        ROC.append(0)
        
        Mi.append((high[i]+low[i]+close[i])/3)
        
    for i in range(11):
        EMA12.append(0)
    
    for i in range(25):
        EMA26.append(0)
        DIFF.append(0)
        MACD.append(0)
        
    stoD.append(0)
    stoD.append(0)    
    ado.append(0)
    Up.append(0)
    Dn.append(0)
    
    for i in range(9,len(high)):
        x = high[i-9:i+1]
        y = low[i-9:i+1]
        z = close[i-9:i+1]
        ans1=maxa(x)
        ans2=mina(y)
        ans3=(close[i]-ans2)/(ans1-ans2)*100
        ans4=avg(z)
        ans5=weigavg(z)
        Hh.append(ans1)
        Ll.append(ans2)
        stoK.append(ans3)
        simavg.append(ans4)
        weiavg.append(ans5)
        
        Mi.append((high[i]+low[i]+close[i])/3)
        SMi.append(avg(Mi[i-9:i+1]))
        M = Mi[i-9:i+1]
        ROC.append((close[i]-close[i-9])/close[i-9])
        sum1=0
        for j in range(len(M)):
            sum1 = sum1 + abs(M[j]-SMi[i])
            
        Di.append(sum1)
        CCI.append((Mi[i]-SMi[i])/0.015*Di[i])
        WR.append((ans1-close[i])/(ans1-ans2)*-100)
            
        if(i>=11):
            d = stoK[i-2:i]
            stoD.append(avg(d))
    
    for i in range(11,len(close)):
        x = avg(close[i-11:i+1])
        if(i==11):
            EMA12.append(x)
        else:    
            EMA12.append(close[i]*(2/13) + EMA12[i-1]*(1-(2/13)))
            
    for i in range(25,len(close)):
        x = avg(close[i-25:i+1])
        y = avg(close[i-9:i+1])
        if(i==25):
            EMA26.append(x)
        else:    
            EMA26.append(close[i]*(2/27) + EMA26[i-1]*(1-(2/27)))
        
        DIFF.append(EMA12[i]-EMA26[i])
        if(i==25):
            MACD.append(y)
        else:    
            MACD.append(MACD[i-1] + 2/11*(DIFF[i]-MACD[i-1]))
    
    
    for i in range(1,len(close)):
        t = close[i]-close[i-1]
        
        if(t>0):
            Up.append(t)
            Dn.append(0)
            label.append(1)
        else:
            Up.append(0)
            Dn.append(-t)
            label.append(-1)
    
    for i in range(9,len(high)):
        u=Up[i-9:i+1]
        d=Dn[i-9:i+1]
        rsi.append(100-(100/(1+avg(u)/avg(d))))
        
    for i in range(1,len(close)):
        ans = (high[i]-close[i-1])/(high[i]-low[i])
        ado.append(ans)

    label.append(0)
    X=[]
    y=[]
    X.append(simavg)
    X.append(weiavg)
    X.append(ROC)
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
    y = np.array(y)
    y = y.transpose()
    
    return X,y

