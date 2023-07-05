import copy
import cvxpy as cp
import collections
import exceptions
import numpy as np
from exceptions.Exception import OptimizationError, InstantiationError
import scipy.optimize as sco
from tools.Loader import *
from data.Tree import *
from BaseOptimizer import *

def get_covariance_matrix(nodes: List[Node])->np.ndarray:
    """
    List로 정렬된 Node들의 집합을 전달받아 공분산 행렬을 반환한다.
    이 때 사용되는 node의 parmas에 저장된 covariance는 한개의 Node과 다른 모든 자산간의 공분산 pd.Series이다.
    selected_covariances를 생성하는 이유는 앞의 cov_series에서 모든 자산군간의 공분산에서 연산에 필요한 행렬만 추출하기 위해서다.
    """

    cov_dict = {}

    node_names = [node.name for node in nodes]  # Get the names of the nodes in the list
    
    for node in nodes:
        if 'covariance' in [i for i in node.params.keys()]:
            pass
        else:
            print('공분산 정보가 없습니다')
            return None

    for node in nodes:
        cov_series = node.params['covariance']
        selected_covariances = cov_series[cov_series.index.isin(node_names)]
        cov_dict[node.name] = selected_covariances

    cov_matrix = pd.DataFrame(cov_dict)

    cov_matrix = cov_matrix.T.fillna(cov_matrix).T

    cov_matrix = (cov_matrix + cov_matrix.T) / 2

    return cov_matrix

def random_covariance_matrix(nodes: List[Node])->np.ndarray:
    """
    node params에 공분산 정보가 없더라도 프로그램이 작동되게 만들기 위해서 랜덤하게 공분산 행렬을 생성하는 역할을 한다.
    """
    size = len(nodes)
    rnd = np.random.rand(size, size)
    return np.matmul(rnd, rnd.T)

def risk_parity_optimizer(nodes: List[Node], covariance_matrix: np.ndarray = None, weight_bounds: Union[List[Tuple], Tuple] = (0, 1)) -> List[float]:
    
    def risk_parity_objective(w, covariance_matrix, eps=1e-8):
        portfolio_variance = w.T @ covariance_matrix @ w
        asset_contributions = w * np.diag(covariance_matrix @ w)
        log_asset_contributions = np.log(asset_contributions + eps)
        return np.sum((log_asset_contributions - log_asset_contributions.mean())**2)
    
    covariance_matrix = get_covariance_matrix(nodes)
    #print(covariance_matrix)
    
    if isinstance(covariance_matrix, np.ndarray):
        pass
    elif isinstance(covariance_matrix, type(None)):
        covariance_matrix = random_covariance_matrix(nodes)

    n_assets = covariance_matrix.shape[0]

    # Create the optimizer
    opt = BaseConvexOptimizer(
        n_assets=n_assets,
        tickers=[node.name for node in nodes],
        weight_bounds=weight_bounds,
    )

    # Use nonconvex_objective method with the risk_parity_objective function
    opt.nonconvex_objective(
        risk_parity_objective,
        objective_args=(covariance_matrix,),
        weights_sum_to_one=True,
    )

    return list(opt.clean_weights().values())


def black_litterman_optimizer(
    nodes: List[Node],
    covariance_matrix: Optional[np.ndarray] = None,
    weight_bounds: Union[List[Tuple], Tuple] = (0, 1),
    tau: float = 0.05,
    P: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    delta: float = 1,
) -> List[float]:

    def black_litterman_objective(w, mu, covariance_matrix, tau, P, Q, omega, delta):
        w_eq = np.linalg.inv(delta * covariance_matrix) @ mu
        pi = delta * covariance_matrix @ w_eq
        sigma = covariance_matrix + tau * P.T @ np.linalg.inv(omega) @ P
        mu_bl = np.linalg.inv(np.linalg.inv(tau * covariance_matrix) + P.T @ np.linalg.inv(omega) @ P) @ (np.linalg.inv(tau * covariance_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        return (w - mu_bl).T @ sigma @ (w - mu_bl)
    
    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    expected_returns = np.array([node.params['returns'] for node in nodes])

    # Create the optimizer
    opt = BaseConvexOptimizer(n_assets=n_assets, tickers=tickers, weight_bounds=weight_bounds)

    covariance_matrix = get_covariance_matrix(nodes)
    #print(covariance_matrix)
    
    if isinstance(covariance_matrix, np.ndarray):
        pass
    elif isinstance(covariance_matrix, type(None)):
        covariance_matrix = random_covariance_matrix(nodes)

    # Set up views and associated matrices if not provided
    if P is None or Q is None:
        # Default: neutral view, no additional views
        P = np.eye(n_assets)
        Q = expected_returns

    if omega is None:
        omega = P @ covariance_matrix @ P.T * tau

    opt.nonconvex_objective(
        black_litterman_objective,
        objective_args=(expected_returns, covariance_matrix, tau, P, Q, omega, delta),
        weights_sum_to_one=True
    )

    return list(opt.clean_weights().values())

def test_optimizer(nodes: List[Node], weight_bounds: Union[List[Tuple], Tuple] = (0, 1)) -> List[float]:
    
    def mean_return(weights, expected_returns):
        return weights @ expected_returns

    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    expected_returns = np.array([node.params['returns'] for node in nodes])

    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)

    result = optimizer.convex_objective(lambda w: -mean_return(w, expected_returns), weights_sum_to_one=True)

    weights = list(result.values())
    return weights


def equal_weight_optimizer(nodes: List[Node], weight_bounds: Union[List[Tuple], Tuple] = (0, 1)) -> List[float]:
    n = len(nodes)
    return [1.0 / n] * n

# covariance_matrix 없어서 안됨. 우선 ticker을 기준으로 데이터끌어와서 covariacne_matrix, expected_return, variance return 만들어야 할듯
def mean_variance_optimizer(nodes: List[Node], covariance_matrix: Optional[np.ndarray] = None, weight_bounds: Union[List[Tuple], Tuple] = (0, 1), risk_aversion: float = 1.0) -> List[float]:
    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    expected_returns = np.array([node.params['returns'] for node in nodes])
    
    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)
    covariance_matrix = get_covariance_matrix(nodes)
    #print(covariance_matrix)
    
    if isinstance(covariance_matrix, np.ndarray):
        pass
    elif isinstance(covariance_matrix, type(None)):
        covariance_matrix = random_covariance_matrix(nodes)

    optimizer.convex_objective(
        lambda w: risk_aversion * portfolio_variance(w, covariance_matrix) - mean_return(w, expected_returns),
        weights_sum_to_one=True
    )

    weights = list(optimizer.clean_weights().values())
    return weights

def portfolio_variance(weights, covariance_matrix):
    return cp.quad_form(weights, covariance_matrix)

def mean_return(weights, expected_returns):
    return weights @ expected_returns