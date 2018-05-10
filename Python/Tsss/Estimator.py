# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:43:34 2018

@author: szkfzx
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Signal
import Trader

class Estimator:
    """
      画出
      signal_data is Pandas
      tran_data is a list of dict
    """
    def draw_trader(signal_data, tran_data):
        ## Touch a figure
        fig = plt.figure()

        ax1 = fig.add_subplot(1,1,1)
    
        ## Paint basic price graphic
        signal_data[['close','slow','fast','filter']].plot(ax=ax1)
        
        ##
        tran_data = pd.DataFrame(tran_data)
        tran_data.set_index(pd.to_datetime(tran_data['date']), inplace=True)
        
        ## Annotate List
        note_data =[]

        ## Generate Annotate Data
        tran_data['tag'] = tran_data['tag'].apply(lambda x:np.where(x=='B-2', 'red', np.where(x=='S-2','green','none')))
        
        for index,row in tran_data.iterrows():
            if row['tag'] == 'none':
                pass
            else:
                note_data.append([index,row['price'],row['tag'],np.round(row['amount'],2)])

        ## Paint Annotates
        for x,y,color,label in note_data:
            ax1.annotate(label,
                     xy = (x, y + 10),
                     xytext = (x, y + 30),
                     arrowprops = dict(facecolor=color,headwidth=7.5,headlength=7.5,width=5),
                     horizontalalignment='left',
                     verticalalignment='top')

if __name__ == '__main__':
    dateParser = lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')
    price = pd.read_csv('data//sh000001.day.csv', 
                        sep=',',
                        header=None,
                        names=['id','date','open','high', 'low', 'close', 'money', 'volume','rate'],
                        index_col=['date'],
                        parse_dates=['date'],
                        date_parser=dateParser)
    dma_signal = Signal.DmaSignal(slow_ma=50, fast_ma=10, filter_ma=222)
    dma_trader = Trader.DmaTrader(init_money = 10000)
    
    signal_data = dma_signal.touch(price)
    dma_trader.run(signal_data)
    
    dma_estimator = Estimator()
    Estimator.draw_trader(signal_data, dma_trader.get_tran_data())