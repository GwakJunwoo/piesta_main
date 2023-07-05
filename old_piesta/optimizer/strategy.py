import pandas as pd
import numpy as np
from piesta.data.asset import Universe
from piesta.optimizer.optimize import MeanVarianceOptimizer, GoalBasedOptimizer, BlackLittermanOptimizer, NaiveOptimizer, RiskParityOptimizer, CustomOptimizer

def mean_variance_strategy(universe: Universe, data: pd.DataFrame) -> pd.Series:
    assets = universe.get_last_layer()
    optimizer = MeanVarianceOptimizer()
    return optimizer.optimize(data[assets])