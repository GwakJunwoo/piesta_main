from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from copy import deepcopy
import pandas as pd
import numpy as np
from Loader import Loader
from Calendar import TradingDayCalendar

class AssetAssumption:
    def __init__(self, Calendar: Optional[TradingDayCalendar]=None,**params: Any):
        self.param_dict = params
        self.param_key = [key for key in params.keys()]
        self.calendar = Calendar

        self.param = {}
        self.max_window = 0

    def set_calendar(self, calendar:TradingDayCalendar):
        self.calendar = calendar
        
    def get_data(self, prices):
        # Extract the maximum window from the input parameters
        self.compute_parameters(prices)
        
        return self.param
        
    def compute_parameters(self, price_data):
        #weekly_price = deepcopy(price_data.resample('w').last())
        #daily_rtn = price_data.pct_change(periods=5).dropna()
        for keys in self.param_key:
            func, kwargs = self.param_dict[keys]
            self.param[keys] = func(self.calendar, price_data, **kwargs)


def historical_return(daily_rtn_data:Optional[pd.DataFrame]=None, window=4):
    if isinstance(daily_rtn_data, pd.DataFrame):
        historical_return = daily_rtn_data.rolling(window=window).mean() * 52 * 100
        return historical_return.dropna()
    

def historical_variance(daily_rtn_data:Optional[pd.DataFrame]=None, window=4):
    if isinstance(daily_rtn_data, pd.DataFrame):
        historical_var = daily_rtn_data.rolling(window=window).std() * np.sqrt(52)
        return historical_var.dropna()

def historical_covariance(Calendar: TradingDayCalendar, daily_rtn_data:Optional[pd.DataFrame]=None, window=4):
    daily_rtn_data = daily_rtn_data.pct_change()
    if isinstance(daily_rtn_data, pd.DataFrame):
        historical_cov = daily_rtn_data.rolling(window=window).cov() * np.sqrt(52) * 100
        return historical_cov.dropna()
    
"""def historical_weekly_return(Calendar: TradingDayCalendar, price_df: Optional[pd.DataFrame]=None, window=4):
    returns = []
    for index in price_df.index:
        start_date = Calendar.find_previous_trading_day(index - pd.DateOffset(weeks=window), price_df.index)
        end_date = index
        trading_days = Calendar.get_all_trading_days(start_date, end_date)

        if len(trading_days) < window * 5:
            returns.append(pd.NA)
        else:
            start_price = price_df.loc[start_date]
            end_price = price_df.loc[end_date]
            returns.append((((end_price / start_price) - 1) * 52))

    avg_returns = pd.DataFrame(returns, index=price_df.index, columns=price_df.columns)
    return avg_returns"""

def historical_weekly_return(Calendar: TradingDayCalendar, price_df: Optional[pd.DataFrame] = None, window=4):
    returns = pd.DataFrame(index=price_df.index, columns=price_df.columns)  # Initialize an empty DataFrame

    for column in price_df.columns:
        column_returns = []
        for index in price_df.index:
            start_date = Calendar.find_previous_trading_day(index - pd.DateOffset(weeks=window), price_df.index)
            end_date = index
            trading_days = Calendar.get_all_trading_days(start_date, end_date)

            if len(trading_days) < window * 5:
                column_returns.append(pd.NA)
            else:
                start_price = price_df.loc[start_date, column]
                end_price = price_df.loc[end_date, column]
                column_returns.append((((end_price / start_price) - 1) * 52))

        returns[column] = column_returns

    print(returns)
    return returns

def historical_weekly_covariance(Calendar: TradingDayCalendar, price_df: Optional[pd.DataFrame] = None, window=4):
    returns = price_df.pct_change().dropna()

    covariances = []
    for index in returns.index:
        start_date = Calendar.find_previous_trading_day(index - pd.DateOffset(weeks=window), price_df.index)
        end_date = index
        trading_days = Calendar.get_all_trading_days(start_date, end_date)

        if len(trading_days) < window * 5:
            pass
        else:
            start_returns = returns.loc[start_date:end_date]
            covariance = start_returns.cov()
            covariances.append(covariance)

    all_covariances = pd.concat(covariances, keys=price_df.index, names=['Date'])

    print(all_covariances.index)
    print(all_covariances.columns)
    return all_covariances

    
"""
assumption = AssetAssumption(returns=(historical_return, {'window': 20}), 
                             covariance=(historical_covariance, {'window': 60}))
"""



loader = Loader("./sample_2.csv")
