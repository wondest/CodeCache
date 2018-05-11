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
        
    def run(self):
        print("AbstractTrader.run")

    def get_tran_data(self):
        print("AbstractTrader.get_tran_data")
   
    def get_acct_data(self):
        print("AbstractTrader.get_acct_data")
        
class DmaTrader(AbstractTrader):
    def __init__(self, init_money):
        AbstractTrader.__init__(self)
        
        ## 设置初始资金
        self.cash_account.buy_capital(init_money)
        
        ##
        self.tran_data = []
        
        ##
        self.acct_data = []

    """
      逐日按照买卖信号执行资金交易
    """
    def run(self, signal_data):
        ##
        buySignal = False
        sellSignal = False
        
        ## 
        for index,row in signal_data.iterrows():
            ##
            today = pd.datetime.strftime(index, '%Y/%m/%d')
            
            ## 当日结算
            self._settle(today, stock_rate=row['rate'])
            
            ## 买入卖出的第二天成交
            if buySignal:
                real_money = self._buyInto(row['close'])
                self._add_tran_data(today,'B-2',real_money,row['close'])
                buySignal = False
                
            if sellSignal:
                real_money = self._sellOut(row['close'])
                self._add_tran_data(today,'S-2',real_money,row['close'])
                sellSignal = False
            
            ### 当日出现买入卖出信号,置标识下一日交易
            if self._isBuySignal(row):
                self._add_tran_data(today,'B-1',0,row['close'])
                buySignal = True

            if self._isSellSignal(row):
                self._add_tran_data(today,'S-1',0,row['close'])
                sellSignal = True
                
            ## 收集客户账户信息
            self._gather(today)

    ## 判断买入信号
    def _isBuySignal(self, tran_data):       
        if tran_data['s-buy'] & (tran_data['s-filter'] > 0):
            return True
        else:
            return False
    
    ## 判断卖出信号
    def _isSellSignal(self, tran_data):
        if tran_data['s-sell']:
            return True
        else:
            return False
    
    ## 执行买入操作
    def _buyInto(self, price):
        ## 取出现金    
        plan_amt = self.cash_account.draw(self._calBuyAmount())
        
        ## 买入股票
        if plan_amt > 0:
            self.stock_account.buy_capital(plan_amt, price)
 
        return plan_amt
    
    ## 计算买入金额
    def _calBuyAmount(self):
        return self.cash_account.get_bal()
        
    ## 执行卖出操作
    def _sellOut(self, price):
        plan_units = self._calSellAmount()
           
        if (plan_units > 0) & (self.stock_account.sell_units(plan_units, price) > 0):
            self.cash_account.save(plan_units*price)
        else:
            plan_units = 0

        return plan_units*price
    
    ## 计算卖出金额
    def _calSellAmount(self):
        return self.stock_account.get_units()
    
    ## 执行账户结算
    def _settle(self, tran_date, **kwargs):
        stock_rate = kwargs['stock_rate']
        
        self.stock_account.settle(tran_date, stock_rate)
  
    ## 汇总账户数据
    def _gather(self, tran_date):
        self.cust_account.gather(tran_date)
        
        #self.tran_data.append(self.cust_account.get_detail())
        self.acct_data.append(self.cust_account.get_detail())
    
    ## 添加交易数据
    def _add_tran_data(self, tran_date, sign_tag, tran_amount, tran_price):
        self.tran_data.append({'date':tran_date,'tag':sign_tag,'amount':tran_amount,'price':tran_price})

    ## 获取交易数据
    def get_tran_data(self):
        return self.tran_data
   
    ## 获取账户数据
    def get_acct_data(self):
        return self.acct_data
if __name__ == '__main__':
    dateParser = lambda dates:pd.datetime.strptime(dates,'%Y/%m/%d')
    price = pd.read_csv('data//sh000001.day.csv', 
                        sep=',',
                        header=None,
                        names=['id','date','open','high', 'low', 'close', 'volumn1', 'volumn2','rate'],
                        index_col=['date'],
                        parse_dates=['date'],
                        date_parser=dateParser)
    dma_signal = Signal.DmaSignal(slow_ma=50, fast_ma=10, filter_ma=222)
    dma_trader = DmaTrader(init_money = 10000)
    dma_trader.run(dma_signal.touch(price))
    
    tran_data = dma_trader.get_tran_data()
    for row in tran_data:
        print(row)
        
    a = pd.DataFrame(tran_data)
    a.set_index(pd.to_datetime(a['date']), inplace=True, drop=True)