# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:06:18 2018

@author: szkfzx
"""
import pandas as pd
import Account
import Signal

"""
  原始交易数据 => 买卖信号数据
  买卖信号数据 => 资金交易数据 **
  资金交易数据 => 策略评价数据
"""
class AbstractTrader:
    def __init__(self):
        print("AbstractTrader.__init__")
        
        ## 客户账户
        self.cust_account = Account.CustAccount("Polly")
        
        ## 现金账户
        self.cash_account = self.cust_account.add_cash_account('cash')
        
        ## 产品账户列表
        self.stock_account = self.cust_account.add_stock_account('stock')
        
        ## 
        
    def run(self):
        print("AbstractTrader.execute")
        
    def get_detail(self):
        print("AbstractTrader.get_detail")
        
class DmaTrader(AbstractTrader):
    def __init__(self):
        AbstractTrader.__init__(self)

    def run(self, signal_data):        
        ## 
        for index,row in signal_data.iterrows():
            print(row)

    def get_detail(self):
        print("DmaTransaction.get_detail")
if __name__ == '__main__':
    dateParser = lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')
    price = pd.read_csv('data//sh000001.day.csv', 
                        sep=',',
                        header=None,
                        names=['id','date','open','high', 'low', 'close', 'volumn1', 'volumn2','rate'],
                        index_col=['date'],
                        parse_dates=['date'],
                        date_parser=dateParser)
    dma_signal = Signal.DmaSignal(slow_ma=100, fast_ma=10, filter_ma=222)
    dma_trader = DmaTrader()
    ##dma_estimator = 
    dma_trader.run(dma_signal.touch(price))
    