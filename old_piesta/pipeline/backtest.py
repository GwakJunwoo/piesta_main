import numpy as np
import pandas as pd
from typing import List, Callable, Dict
from datetime import datetime, timedelta

class Backtest:
    def __init__(self, pipeline: AssetAllocationPipeline, data: pd.DataFrame, start_date: datetime, end_date: datetime, rebalancing_frequency: timedelta, transaction_cost: float = 0.0, slippage: float = 0.0, benchmark: pd.DataFrame = None):
        self.pipeline = pipeline
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.benchmark = benchmark

    def run_backtest(self) -> Dict:
        backtest_results = {}
        current_date = self.start_date
        weights = None
        assets = self.pipeline.universe.get_last_layer()
        rebalancing_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalancing_frequency)

        while current_date <= self.end_date:
            date_data = self.data.loc[self.data.index == current_date]
            if not date_data.empty:
                if current_date in rebalancing_dates or weights is None:
                    allocation_results = self.pipeline.run_pipeline(date_data)
                    new_weights = allocation_results[f'L{len(allocation_results)}']

                    cost = self._calculate_cost(weights, new_weights) if weights is not None else 0
                    weights = new_weights

                weighted_returns = self._calculate_weighted_returns(date_data, assets, weights, cost)
                backtest_results[current_date] = weighted_returns.values[0]

            current_date += timedelta(days=1)

        backtest_df = self._generate_backtest_dataframe(backtest_results)
        return self._generate_results(backtest_df)

    def _calculate_cost(self, weights: pd.Series, new_weights: pd.Series) -> float:
        turnover = np.abs(weights - new_weights).sum() / 2
        return turnover * (self.transaction_cost + self.slippage)

    def _calculate_weighted_returns(self, date_data: pd.DataFrame, assets: List[str], weights: pd.Series, cost: float) -> pd.Series:
        asset_returns = date_data[assets].pct_change()
        return asset_returns.mul(weights).sum() - cost

    def _generate_backtest_dataframe(self, backtest_results: Dict) -> pd.DataFrame:
        backtest_df = pd.Series(backtest_results)
        if self.benchmark is not None:
            benchmark_returns = self.benchmark.loc[self.start_date:self.end_date].pct_change()
            backtest_df = pd.concat([backtest_df, benchmark_returns], axis=1, join='inner')
            backtest_df.columns = ['Strategy', 'Benchmark']
            backtest_df['Excess Returns'] = backtest_df['Strategy'] - backtest_df['Benchmark']
            backtest_df = backtest_df.dropna()
        return backtest_df

class GenerateResults:
    def __init__(self, backtest_df: pd.DataFrame):
        self.backtest_df = backtest_df

    def generate_results(self) -> Dict:
        results = {}
        results['Returns'] = self.backtest_df['Strategy'].sum()
        results['Volatility'] = self.backtest_df['Strategy'].std() * np.sqrt(252)
        results['Sharpe Ratio'] = results['Returns'] / results['Volatility']
        results['Max Drawdown'] = (self.backtest_df['Strategy'].cummax() - self.backtest_df['Strategy']).max()
        results['Turnover'] = (np.abs(self.backtest_df['Strategy'].diff()) / 2).mean() / self.backtest_df['Strategy'].mean()

        if 'Benchmark' in self.backtest_df.columns:
            results['Benchmark Returns'] = self.backtest_df['Benchmark'].sum()
            results['Excess Returns'] = self.backtest_df['Excess Returns'].sum()
            results['Information Ratio'] = results['Excess Returns'] / self.backtest_df['Excess Returns'].std()
            results['Tracking Error'] = self.backtest_df['Excess Returns'].std()

        return results
