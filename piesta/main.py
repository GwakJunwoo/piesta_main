import warnings
from Loader import *
from Optimizer import *
from Pipeline import *
from Assumption import *
from Backtest import *
from Calendar import TradingDayCalendar

warnings.filterwarnings(action='ignore')

# Load asset prices
loader = Loader("./sample_2.csv")

universe = Tree("Universe")
universe.insert("Universe", "ACWI", weight_bounds = (0.1, 0.8))
universe.insert("Universe", "BOND", weight_bounds = (0.1, 0.8))
universe.insert("Universe", "REET", weight_bounds = (0, 0.5))
universe.insert("Universe", "USO", weight_bounds = (0, 0.5))
universe.insert("Universe", "GLO", weight_bounds = (0, 0.5))
universe.insert("Universe", "JPST", weight_bounds = (0, 0.2))

universe.insert("ACWI", "SPY", weight_bounds = (0.1, 0.6))
universe.insert("ACWI", "EWY", weight_bounds = (0.1, 0.6))

universe.insert("BOND", "GOVT", weight_bounds = (0.1, 0.6))
universe.insert("BOND", "LQD", weight_bounds = (0.1, 0.6))

universe.insert("SPY", "VTV", weight_bounds = (0.1, 0.7))
universe.insert("SPY", "VUG", weight_bounds = (0.1, 0.7))
universe.insert("SPY", "VYM", weight_bounds = (0.1, 0.7))
universe.insert("SPY", "MTUM", weight_bounds = (0.1, 0.7))
universe.insert("SPY", "SPLV", weight_bounds = (0.1, 0.7))
universe.insert("SPY", "QUAL", weight_bounds = (0.1, 0.7))

universe.insert("EWY", "325010", weight_bounds = (0.1, 0.7))
universe.insert("EWY", "223190", weight_bounds = (0.1, 0.7))
universe.insert("EWY", "325020", weight_bounds = (0.1, 0.7))

universe.insert("GOVT", "SHV", weight_bounds = (0.1, 0.7))
universe.insert("GOVT", "TLH", weight_bounds = (0.1, 0.7))

universe.insert("LQD", "VCSH", weight_bounds = (0.4, 0.7))
universe.insert("LQD", "VCLT", weight_bounds = (0.4, 0.7))

universe.insert("GLO", "GLO", weight_bounds = (0.4, 0.6))
universe.insert("GLO", "SLV", weight_bounds = (0.4, 0.6))

universe.insert("REET", "SRVR", weight_bounds = (0.4, 0.6))
universe.insert("REET", "XHB", weight_bounds = (0.4, 0.6))

"""universe = Tree("Universe")
universe.insert("Universe", "ACWI")
universe.insert("Universe", "BOND")
universe.insert("Universe", "REET")
universe.insert("Universe", "USO")
universe.insert("Universe", "GLO")

universe.insert("ACWI", "SPY")
universe.insert("ACWI", "EWY")

universe.insert("BOND", "GOVT")
universe.insert("BOND", "LQD")

universe.insert("SPY", "VTV")
universe.insert("SPY", "VUG")
universe.insert("SPY", "VYM")
universe.insert("SPY", "MTUM")
universe.insert("SPY", "SPLV")
universe.insert("SPY", "QUAL")

universe.insert("EWY", "325010")
universe.insert("EWY", "223190")
universe.insert("EWY", "325020")

universe.insert("GOVT", "SHV")
universe.insert("GOVT", "TLH")

universe.insert("LQD", "VCSH")
universe.insert("LQD", "VCLT")

universe.insert("GLO", "GLO")
universe.insert("GLO", "SLV")

universe.insert("REET", "SRVR")
universe.insert("REET", "XHB")"""


pipe = pipeline([("SAA", risk_parity_optimizer), ("TAA", mean_variance_optimizer), ("AP", risk_parity_optimizer)], universe)
assumption = AssetAssumption(returns=(historical_weekly_return, {'window': 4}), covariance=(historical_weekly_covariance, {'window': 4}))

# Create a Backtest object
backtest = Backtest(pipeline=pipe,
    loader=loader,
    assumption=assumption,
    start_date="2020-01-07",
    end_date="2023-05-01",
)


# Run the backtest
backtest.run_backtest()

# Plot the performance
backtest.plot_performance()

# Calculate maximum drawdown and turnover
max_drawdown = backtest.calculate_maximum_drawdown()
turnover = backtest.calculate_turnover()
asset_weights = backtest.asset_weights
returns = backtest.calculate_return()

print("Max Drawdown:", max_drawdown)
print("Turnover:", turnover)
print("returns:", (returns[-1]-1)*100)
asset_weights