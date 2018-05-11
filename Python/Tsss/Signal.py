# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:40:44 2018

@author: xieqin

   输入：原始交易数据
   数据：带买入卖出信号数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
  原始交易数据 => 买卖信号数据 **
  买卖信号数据 => 资金交易数据
  资金交易数据 => 策略评价数据
"""
class AbstractSignal:
    ## 初始化函数
    def __init__(self):
        print("AbstractSignal.__init__")
        pass

    ## 指标信号
    def _touch_indicate(self,input_data):
        print("AbstractSignal._touch_indicate")
        return input_data
    
    ## 信号
    def _append_signal(self, input_data):
        print("AbstractSignal._touch_signal")
        return input_data
    
    ## 过滤
    def _append_filter(self, input_data):
        print("AbstractSignal._append_filter")
        return input_data

    def touch(self, input_data):
        ##
        mid_data1 = self._touch_indicate(input_data)
        ##
        mid_data2 = self._append_signal(mid_data1)
        ##
        signal_data = self._append_filter(mid_data2)
        ##
        return signal_data
 
"""
  双均线策略：
    1. 短期均线上穿长期均线，视为买入信号；
    2. 短期均线下穿长期均线，视为卖出信号；
"""
class DmaSignal(AbstractSignal):
    ## 初始化函数
    def __init__(self, slow_ma, fast_ma, filter_ma):
        ##
        AbstractSignal.__init__(self)
        ## 慢均线
        self.slow_ma = slow_ma
        ## 快均线
        self.fast_ma = fast_ma
        ## 趋势过滤
        self.filter_ma = filter_ma
    
    ## 创建指标信号
    def _touch_indicate(self,input_data):
        ## 收盘价格
        close = input_data['close']
        
        ## 收益率
        retu_rate = (close-close.shift(1))/close.shift(1)
        
        ## signal 买卖指标 fast/slow
        slow_ma = close.rolling(window=self.slow_ma).mean()
        fast_ma = close.rolling(window=self.fast_ma).mean()
        
        ## filter 顾虑器指标 filter
        filter_ma = close.rolling(window=self.filter_ma).mean()
        
        ## 合并结果
        result = pd.concat([close,retu_rate,slow_ma,fast_ma,filter_ma], keys=['close','rate','slow','fast','filter'], axis=1)
        
        ## 去掉空值
        return result.dropna()
    
    ## 创建
    def _append_filter(self, input_data):
        ## filter 趋势判断 TODO 这个震动区间要 得到一个值, ATR？
        filter_data=(input_data['filter']-input_data['filter'].shift(1)).apply(lambda x:np.where(x<-3,-1,np.where(x>3,1,0)))

        ## filter 趋势判断
        input_data['s-filter'] = filter_data.rolling(window=10).mean().apply(lambda x:np.where(x<-0.5,-1,np.where(x>0.5,1,0)))

        return input_data.dropna()   
    
    ## 创建买卖信号
    def _append_signal(self, input_data):
        ##
        diff = input_data['fast'] - input_data['slow']
        
        ## fast 向上突破 发出买入信号
        input_data['s-buy'] = (diff > 0) & (diff.shift(1) <= 0)
        
        ## fast 向下突破 发出卖出信号
        input_data['s-sell'] = (diff < 0) & (diff.shift(1) >= 0)
        
        return input_data.dropna()

if __name__ == '__main__':
    dateParser = lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')
    price = pd.read_csv('data//sh000001.day.csv', 
                        sep=',',
                        header=None,
                        names=['code','date','open','high', 'low', 'close', 'money', 'volume','rate'],
                        index_col=['date'],
                        parse_dates=['date'],
                        date_parser=dateParser)
    
    test = DmaSignal(slow_ma=100, fast_ma=10, filter_ma=222)
    
    #data = test._touch_indicate(price)
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    data = test.touch(price)
    
    data[['close','filter']].plot(ax=ax1)
    
    ## Annotate List
    note_data =[]

    ## Generate Annotate Data
    for index,row in data.iterrows():
        if row['s-buy']:
            note_data.append([index,row['close'],'red','B'])
        if row['s-sell']:
            note_data.append([index,row['close'],'green','S'])

    ## Paint Annotates
    for x,y,color,label in note_data:
        ax1.annotate(label,
                     xy = (x, y + 10),
                     xytext = (x, y + 30),
                     arrowprops = dict(facecolor=color,headwidth=7.5,headlength=7.5,width=5),
                     horizontalalignment='left',
                     verticalalignment='top')
    
    ## Paint Filter
    data['s-filter'].plot(ax=ax2)
    data['diff'] = (data['filter']-data['filter'].shift(1))
    #data[['filter','diff']].plot()
    data['s-filter']['2015/8':'2015/11'].plot(ax=ax2)
    data['s-filter']['2016/5':'2016/7'].plot(ax=ax2)
    
    data[['s-filter','filter']]['2015/8':'2015/11']