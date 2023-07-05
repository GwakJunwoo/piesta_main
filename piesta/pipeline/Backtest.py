import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime
from Pipeline import *
from tools.Loader import *
from Assumption import *
from tools.Calendar import *
from tools.Calendar import TradingDayCalendar


class Backtest:
    def __init__(
        self,
        pipeline: pipeline,
        loader: Loader,
        assumption: AssetAssumption,
        start_date: str,
        end_date: str,
        rebalancing_frequency: str = "1m", # 반영안되고 있음 커스텀비즈니스 캘린더를 만들던가 해야함 
        rebalancing_fee: float = 0.001,
        #calendar: Calendar = Calendar()
    ):
        self.pipeline = pipeline
        self.loader = loader
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        #self.calendar = calendar
        self.rebalancing_frequency = rebalancing_frequency
        self.rebalancing_fee = rebalancing_fee
        self.cl = TradingDayCalendar(self.start_date, self.end_date)
        self.assumption = assumption
        self.assumption.set_calendar(self.cl)
    
        
    def previous_business_day(self, date: datetime) -> datetime:
        date_series = pd.date_range(end=date, periods=2, freq='B')
        return date_series[0]

    def _generate_rebalancing_dates(self, date_index):
        # rebalancing_dates은 비즈니스 데이를 가정하지 않음 그래서 일치하지 않는다는 오류가 발생함.
        # 애초에 이게 Calendar 모듈로 빠져야함
        # CBM 해도 없으면 또 누락됨 ㅇㅇ...
        rebalancing_dates = self.cl.get_month_end_trading_days(self.start_date, self.end_date, freq=self.rebalancing_frequency)
        return self.cl.find_matching_dates(date_index, rebalancing_dates)

    def run_backtest(self):
        prices = self.loader.load_data(self.cl.get_start_date(), self.cl.get_end_date())
        assumption_dict = self.assumption.get_data(prices=prices)
        
        # def _update_node(self, assumption_dict:Dict, dates:str)
        self.pipeline._update_node(assumption_dict, self.start_date)
        
        rebalancing_dates = self._generate_rebalancing_dates(prices.index)

        # Include the start date as a rebalancing date
        rebalancing_dates = rebalancing_dates.insert(0, self.start_date)

        asset_weights = []
        portfolio_value = [1]

        print("=========================================================")
        for i, rebalancing_date in enumerate(rebalancing_dates):
            #prices_sub = prices.loc[:rebalancing_date]
            # 자산 assumption 업데이트
            #assumption_dict = self.assumption.get_data(self.loader, self.start_date, rebalancing_date)
            #print(self.start_date, rebalancing_date)
            self.pipeline._update_node(assumption_dict, rebalancing_date)
            #print(assumption_dict)
            allocations = self.pipeline.run()

            for asset, weight in allocations.items():
                print(f"{asset}: {weight:.2f}", end=" ")
            print("\n=========================================================")

            asset_weights.append(allocations)

            if i < len(rebalancing_dates) - 1:
                end_date = rebalancing_dates[i + 1]
            else:
                end_date = prices.index[-1]

            start_date = self.previous_business_day(rebalancing_date)
            prices_period = prices.loc[start_date:end_date]
            
            #todo... 왜 2022년 5월 30일이 안잡히냐... ㅅㅂ....
            if start_date < prices_period.index[0]:
                print("ㅈ됨")
                start_date = self.previous_business_day(start_date)
                prices_period = prices.loc[start_date:end_date]
                
            period_returns = prices_period.pct_change().dropna()
            period_returns['portfolio'] = np.dot(period_returns, list(allocations.values()))

            # Apply rebalancing_fee
            if i > 0:
                period_returns.loc[rebalancing_date, 'portfolio'] -= self.rebalancing_fee

            period_cumulative_returns = (1 + period_returns).cumprod()
            period_portfolio_value = portfolio_value[-1] * period_cumulative_returns['portfolio']
            
            if i > 0:
                portfolio_value.extend(period_portfolio_value[1:].values)
            else:
                portfolio_value.extend(period_portfolio_value[:].values)

        self.portfolio_value = pd.Series(portfolio_value, index=prices.index)
        self.returns = self.portfolio_value.pct_change().dropna()
        self.asset_weights = pd.DataFrame(asset_weights, index=rebalancing_dates, columns=prices.columns)

    def plot_performance(self):
        plt.figure(figsize=(12,4))
        plt.plot(self.portfolio_value)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.title("Portfolio Performance")
        plt.show()

    def calculate_maximum_drawdown(self):
        rolling_max = self.portfolio_value.cummax()
        drawdowns = (self.portfolio_value - rolling_max) / rolling_max
        return drawdowns.min()

    def calculate_turnover(self):
        turnover = (self.asset_weights.shift(1) - self.asset_weights).abs().sum(axis=1).mean()
        return turnover
    
    def calculate_return(self):
        return self.portfolio_value