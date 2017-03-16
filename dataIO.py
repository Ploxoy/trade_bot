import numpy as np
import pandas as pd
from datetime import date, datetime


def load_for_LSTM_Bot_by_Dates(filename,time_steps=6):
    
    
    data=np.load(filename)
    
    c=pd.DataFrame(data.f.candles, columns=data.f.candles_names)
    X=np.hstack((data.f.vectors,data.f.deltas))
    categorical_cols = ['Qty','hour', 'minute']
    X=np.hstack((X,c[categorical_cols].applymap(int).values))
    
    dates=  list(set(c["DateTime"].apply(lambda x : x.date())))
    #print (dates)  #TODO проверить тут.
    d = sorted(dates)
    #print(d)  # TODO проверить тут.
    data_byDates=[]
    
    for d1 in d:
        date_indx=c["DateTime"].apply(lambda x : x.date())  ==  d1
        x_d=X[date_indx.index[date_indx]]
        c_d=c[date_indx]
        x3d=go_3d(x_d,time_steps)
        data_byDates.append([d1, x3d, c_d[time_steps-1:] ] ) 
    
    return data_byDates


def load_by_Dates(filename):
    '''загрузка данных без попытки  группировать по временным шагам'''
    data = np.load(filename)

    c = pd.DataFrame(data.f.candles, columns=data.f.candles_names)
    #TODO начисто забыл что такое deltas
    X = np.hstack((data.f.vectors, data.f.deltas))

    #categorical_cols = ['Qty', 'hour', 'minute']
    #X = np.hstack((X, c[categorical_cols].applymap(int).values))

    dates = list(set(c["DateTime"].apply(lambda x: x.date())))
    # print (dates)  #TODO проверить тут.
    d = sorted(dates)
    # print(d)  # TODO проверить тут.
    data_byDates = []

    for d1 in d:
        date_indx = c["DateTime"].apply(lambda x: x.date()) == d1
        x_d = X[date_indx.index[date_indx]]
        c_d = c[date_indx]

        data_byDates.append([d1, x_d, c_d])

    return data_byDates

def go_3d(x_d,time_steps):
    
    len3d=x_d.shape[0]-time_steps+1
    x3d=np.zeros( (len3d,time_steps,x_d.shape[1] ),dtype=x_d.dtype )
    for i in range (len3d):
        x3d[i]=x_d[i:i+time_steps,]
    return x3d  

def load_for_LSTM_by_Dates(filename,time_steps=6,time_end=(19,0) ):
    
    
    data=np.load(filename)
    
    c=pd.DataFrame(data.f.candles,columns=data.f.candles_names)
    
    X_real=np.hstack((data.f.vectors,data.f.deltas))
    X_real=np.hstack((X_real,c.Qty.values.reshape(-1,1)))
   
    categorical_cols = ['hour', 'minute']
    X_cat=c[categorical_cols]
    X_cat=X_cat.applymap(int)
    
   
    
    future=pd.DataFrame(data.f.future, columns=data.f.future_names)
    y=future
    
    X=np.hstack((X_real,X_cat))
    
    dates=  list(set(c["DateTime"].apply(lambda x : x.date())))
    
    data_byDates=[]
    
    for date in dates:
        date_indx=c["DateTime"].apply(lambda x : x.date())  ==  date
        x_d=X[date_indx.index[date_indx]]
        y_d=y[date_indx]
        c_d=c[date_indx]
        x3d=go_3d(x_d,time_steps)
        data_byDates.append([date, x3d, y_d[time_steps-1:] , c_d[time_steps-1:] ] ) 
    
    return data_byDates