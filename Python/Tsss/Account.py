# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:03:07 2018

@author: szkfzx
"""

class InnerAccount:
    def __init__(self):
        ##持有数量
        self.hold_units=0.00
        
        ##持有单价
        self.hold_price=1
    """
      增加单元
    """
    def increase(self, units, price=1):
        now_cost = units*price
        old_cost = self.get_cost()
        
        self.hold_units += units
        if self.hold_units == 0:
            self.hold_price = 0
        else:
            self.hold_price = (old_cost+now_cost)/self.hold_units

        return units
    """
      减少单元
    """
    def decrease(self, units, price=1):
        now_cost = units*price
        old_cost = self.get_cost()
        
        self.hold_units -= units
        if self.hold_units == 0:
            self.hold_price = 0
        else:
            self.hold_price = (old_cost-now_cost)/self.hold_units
    
        return units
    """
      持有成本
    """
    def get_cost(self):
        return self.hold_units*self.hold_price

    """
      持有数量
    """
    def get_units(self):
        return self.hold_units

    """
      持有单价
    """
    def get_price(self):
        return self.hold_price

"""
  基础账户
"""
class BaseAccount:
    def __init__(self):
        self.inner_account = InnerAccount()
        
        ## 当日盈亏
        self.daily_profit = 0.00
        
        ## 累计盈亏
        self.total_profit = 0.00
        
        ##
        self.last_deal_price = 0.00
        self.last_deal_date='1899/12/31'
        self.last_deal_rate=0.00
        
    def is_zero(self):
        return (self.inner_account.get_cost() <= 0)
    
    def get_units(self):
        return self.inner_account.get_units()

    def get_daily_profit(self):
        return self.daily_profit
        
    def get_total_profit(self):
        return self.total_profit
    
    """
      买入单元: 金额买入
    """
    def buy_capital(self, capital, price=1):
        if capital<= 0 or price <= 0:
            return -1
        self.buy_units(capital/price, price)
    
    """
      买入单元: 份额买入
    """
    def buy_units(self, units, price):
        if units <= 0:
            return -1
        
        self.last_deal_price = price
        return self.inner_account.increase(units, price)

    """
      卖出单元: 份额卖出
    """
    def sell_units(self, units, price):
        if units <= 0:
            return -1

        if units > self.get_units():
            return 0

        self.last_deal_price = price
        return self.inner_account.decrease(units, price)

    ## 
    def get_bal(self):
        print("BaseAccount.get_bal")
        
    ## 
    def settle(self, tran_date, rate):
        print("BaseAccount.settle")
        
    ##
    def get_detail(self):
        print("BaseAccount.get_detail")

"""
  资金账户
  1. 不允许透支
  2. 只允许存入>0的值
  3. 只允许提取>0的值
"""
class CashAccount(BaseAccount):
    def __init__(self):
        ##持有单元
        BaseAccount.__init__(self)
    
    def draw(self, plan_amt):
        if plan_amt <= 0:
            return -1
        
        if plan_amt > self.get_bal():
            return 0
        
        self.inner_account.decrease(plan_amt)
        return plan_amt;
    
    def save(self, plan_amt):
        if plan_amt <= 0:
            return -1

        self.inner_account.increase(plan_amt)
        return plan_amt;
      
    def get_bal(self):
        return self.inner_account.get_cost()
    
"""
  货币基金账户
"""
class MonetaryAccount(BaseAccount):
    def __init__(self):
        ##持有单元
        BaseAccount.__init__(self)
    """
      持有市场价值
    """
    def get_bal(self):
        return self.inner_account.hold_units*self.last_deal_price
    
    """
      结算: 按收益率结算，红利再投入
    """
    def settle(self, tran_date, rate):
        ##
        self.daily_profit = self.get_bal()*rate
        self.total_profit += self.daily_profit
        
        ##
        self.last_deal_date = tran_date
        self.last_deal_rate = rate
        
        ##
        self.buy_capital(self.daily_profit, 1)
        
    """
      返回持仓信息
    """
    def get_detail(self):
        return {'date':self.last_deal_date,'balance':self.get_bal(),'daily_profit':self.get_daily_profit(),'total_profit':self.get_total_profit(),'hold_units':self.inner_account.hold_units,'rate':self.last_deal_rate}

    
"""
  股票基金账户
"""
class StockAccount(BaseAccount):
    def __init__(self):
        ##持有单元
        BaseAccount.__init__(self)

    """
      持有市场价值
    """
    def get_bal(self):
        return self.inner_account.hold_units*self.last_deal_price
    
    """
      账户结算：
        按单位净值结算：股票基金
    """
    def settle_price(self, tran_date, price):
        ##
        rate = (price - self.last_deal_price)/self.last_deal_price
        
        ##
        self.settle(tran_date, rate)

    """
      账户结算：
        按日收益率结算：货币基金
    """ 
    def settle(self, tran_date, rate):
        ##
        price = self.last_deal_price * (1+rate)

        ##
        self.daily_profit = self.get_bal() * rate
        self.total_profit += self.daily_profit
        
        ##
        self.last_deal_rate = rate
        self.last_deal_date = tran_date
        self.last_deal_price = price
        
    """
      持有成本
    """
    def get_cost(self):
        return self.inner_account.get_cost()

    """
      返回持仓信息
    """
    def get_detail(self):
        return {'date':self.last_deal_date,'balance':self.get_bal(),'daily_profit':self.get_daily_profit(),'total_profit':self.get_total_profit(),'hold_units':self.inner_account.hold_units,'hold_price':self.inner_account.hold_price,'price':self.last_deal_price,'rate':self.last_deal_rate}

"""
  客户账户
"""
class CustAccount:
    def __init__(self, name):
        self.name = name
        self.accounts = {}
        
        ##
        self.last_deal_date = '1899/12/31'
        
        self.balance = 0.00
        self.daily_profit = 0.00
        self.total_profit = 0.00
        
        ##最近交易日期
        self.last_deal_date='1899/12/31'
        
    def add_cash_account(self, acct_name):
        new_account = CashAccount()
        self.accounts[acct_name] = new_account
        return new_account
    
    def add_stock_account(self, acct_name):
        new_account = StockAccount()
        self.accounts[acct_name] = new_account
        return new_account

    def add_monetary_account(self, acct_name):
        new_account = MonetaryAccount()
        self.accounts[acct_name] = new_account
        return new_account

    def remove_account(self, acct_name):
        return self.accounts.pop(acct_name)
    
    def get_account(self, acct_name):
        return self.accounts[acct_name]
    
    """
      收集客户名下所有账户的数据
    """
    def gather(self, tran_date):
        self.balance = 0.00
        self.daily_profit = 0.00
        self.total_profit = 0.00
        
        self.last_deal_date = tran_date
        
        for acct_name in self.accounts:
            self.balance += self.accounts[acct_name].get_bal()
            self.daily_profit += self.accounts[acct_name].get_daily_profit()
            self.total_profit += self.accounts[acct_name].get_total_profit()

    def get_bal(self):
        return self.balance

    def get_daily_profit(self):
        return self.daily_profit
        
    def get_total_profit(self):
        return self.total_profit

    """
      返回持仓信息
    """
    def get_detail(self):
        return {'date':self.last_deal_date,'balance':self.get_bal(),'daily_profit':self.get_daily_profit(),'total_profit':self.get_total_profit()}
 
    
if __name__ == '__main__':
    acct = StockAccount()
    
    acct.buy_capital(10000, price=1)
    print(acct.get_detail())
    
    acct.settle('2018/5/10',0.0001)
    print(acct.get_detail())
    
    acct.buy_capital(10000, price=2)
    acct.settle('2018/5/11',0.0001)
    print(acct.get_detail())
    
    acct.settle_price('2018/5/12',0.9)
    print(acct.get_detail())
    
    acct.settle('2018/5/13',0.0003)
    print(acct.get_detail())