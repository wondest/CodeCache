# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:42:14 2018

@author: xnn

  绘制蜡烛图：使用matplotlib.finance 模块中的相关函数进行蜡烛图的绘制
"""

import matplotlib.pyplot as plt
import matplotlib.finance as fin
from datetime import datetime
from matplotlib.dates import date2num,WeekdayLocator,MONDAY,DateFormatter,DayLocator
import pandas as pd
import numpy as np

"""
  输入的是: dataFrame
  
  

"""
def plotCandles(ax, df):
    
    ## time must be in float days format - see date2num
    df['time'] = df.index[:]
    df['time'] = df['time'].apply(lambda x:date2num(x))
    
    alist=list()
    
    ## time, open, high, low, close
    for index,row in df.iterrows():
        alist.append([row['time'],row['open'],row['high'],row['low'],row['close']])
    
    
    """
    """
    mondays = WeekdayLocator(MONDAY)
    weekFormatter = DateFormatter('%y %b %d')
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(weekFormatter)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    """
        sequence of (time, open, high, low, close, ...) sequences
        As long as the first 5 elements are these values,
        the record can be as long as you want (e.g., it may store volume).

        time must be in float days format - see date2num
    """
    fin.candlestick_ohlc(ax, alist, width=0.7, colorup='r', colordown='g')
    
    plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
    
    ax.set_title('上证指数K线图')

"""
    
"""
def catchHopeStars(df):
    for i in range(3,100):
        print(i)

if __name__ == '__main__':
    dateParser = lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')
    origin_data = pd.read_csv('data//sh000001.day.csv', 
                        sep=',',
                        header=None,
                        names=['code','date','open','high', 'low', 'close', 'money', 'volume','rate'],
                        index_col=['date'],
                        dtype={'code': str},
                        parse_dates=['date'],
                        date_parser=dateParser)
    ## Touch a figure
    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)    
    #plotCandles(ax1, origin_data['2013-9':'2013-12'])
    
    
    catchHopeStars(origin_data)