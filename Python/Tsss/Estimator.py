# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:43:34 2018

@author: xieqin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Signal
import Trader

class Estimator:
    def __init__(self, Rf):
        self.Rf = Rf
    
    """
      画出
      signal_data is Pandas
      tran_data is a list of dict
    """
    def draw_trader(signal_data, tran_data, axe):
        ## Paint basic price graphic
        signal_data[['close','slow','fast','filter']].plot(ax=axe)
        
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
            axe.annotate(label,
                     xy = (x, y + 10),
                     xytext = (x, y + 30),
                     arrowprops = dict(facecolor=color,headwidth=7.5,headlength=7.5,width=5),
                     horizontalalignment='left',
                     verticalalignment='top')

    """
      计算各项指标
    """
    def calc_top(self, signal_data, acct_data):
        ## Rm
        head_value = signal_data['close'].head(1)[0]
        tail_value = signal_data['close'].tail(1)[0]
        
        tran_length = len(signal_data)
        
        Rm = (tail_value-head_value)/tail_value/tran_length*365
        
        ## Rp
        acct_data = pd.DataFrame(acct_data)
        acct_data.set_index(pd.to_datetime(acct_data['date']), inplace=True)
        
        head_value = acct_data['balance'].head(1)[0]
        tail_value = acct_data['balance'].tail(1)[0]
        
        Rp = (tail_value-head_value)/tail_value/tran_length*365
        
        ## 波动率
        Vp = acct_data['rate'].std()
        
        ## Beta Alpha
        Comb_data = pd.concat([signal_data['rate'],acct_data['rate']], keys=['Rm','Rp'], axis=1)
        Cov_matrix = Comb_data.cov()
        
        Beta = Cov_matrix['Rm']['Rp']/Cov_matrix['Rm']['Rm']
        
        Alpha = (Rp-self.Rf) - Beta*(Rm-self.Rf)
        
        ## Sharpe Ratio
        SR = (Rp - Rm)/Vp
        
        ## Information Ratio
        IR = (Comb_data['Rp'] - Comb_data['Rm']).mean()/(Comb_data['Rp'] - Comb_data['Rm']).std()
        
        ## 最大回撤
        max_balance = 0.0
        max_drawdown = 0.0
        acct_data['drawdown'] = 0
        
        for index,row in acct_data.iterrows():
            if row['balance'] > max_balance:
                max_balance = row['balance']
            
            row['drawdown'] = (max_balance - row['balance'])/max_balance
            
            if row['drawdown'] > max_drawdown:
                max_drawdown = row['drawdown']
        
        print("Rm=[%f]  Rp=[%f] Beta=[%f] Alpha=[%f] Vp=[%f] SR=[%f] IR=[%f] MaxDD=[%f]" %(Rm, Rp, Beta, Alpha, Vp, SR, IR, max_drawdown))
        
        return {'Rm':Rm,'Rp':Rp,'Beta':Beta,'Alpha':Alpha,'Vp':Vp,'SR':SR,'IR':IR,'MaxDD':max_drawdown}
        
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
    dma_signal = Signal.DmaSignal(slow_ma=50, fast_ma=10, filter_ma=222)
    dma_trader = Trader.DmaTrader(init_money = 10000)
    dma_estimator = Estimator(Rf=0.035)
    
    ## origin_data => signal_data
    signal_data = dma_signal.touch(origin_data)['2015':]
    
    ## signal_data => acct_data,tran_data
    dma_trader.run(signal_data)
    
    tran_data = dma_trader.get_tran_data()
    acct_data = dma_trader.get_acct_data()
    
    ## Touch a figure
    fig = plt.figure()

    axe1 = fig.add_subplot(2,1,1)
    axe2 = fig.add_subplot(2,1,2)
    
    ## Draw
    Estimator.draw_trader(signal_data, tran_data, axe1)
    signal_data[['s-filter']].plot(ax=axe2)
    
    ## Print
    aa = dma_estimator.calc_top(signal_data, acct_data)
    
    print(aa)