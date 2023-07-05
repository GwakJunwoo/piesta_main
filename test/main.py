from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
import collections
import pandas as pd
import json


class Node:
    def __init__(self, name: str, expected_return: float = 0.0, variance: float = 0.0):
        self.name = name
        self.children: List[Node] = []
        self.expected_return = expected_return
        self.variance = variance

    def add_child(self, child_node: 'Node') -> None:
        self.children.append(child_node)

class Tree:
    def __init__(self, root_name: str):
        self.root = Node(root_name)

    def __repr__(self) -> str:
        return f'Tree({self.root.name})'

    def insert(self, parent_name: str, child_name: str, expected_return: float = 0.0, variance: float = 0.0) -> bool:
        parent_node = self._find_node(self.root, parent_name)
        if parent_node:
            child_node = Node(child_name, expected_return, variance)
            parent_node.add_child(child_node)
            return True
        return False

    def draw(self) -> None:
        print(self._build_tree_string(self.root, '')[0])

    def _find_node(self, node: Node, target_name: str) -> Optional[Node]:
        if node.name == target_name:
            return node
        for child in node.children:
            found_node = self._find_node(child, target_name)
            if found_node:
                return found_node
        return None

    def _build_tree_string(self, node: Node, prefix: str, is_tail: bool = True):
        lines = []
        line = f"{prefix}{'`-- ' if is_tail else '|-- '}{node.name}"
        lines.append(line)
        prefix += '    ' if is_tail else '|   '
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = i == child_count - 1
            lines.extend(self._build_tree_string(child, prefix, is_last_child))
        return lines


class Universe:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.level_covariance_matrices = {}

    def insert(self, parent_name: str, child_name: str, **params: Any) -> bool:
        return self.tree.insert(parent_name, child_name, **params)

    def loader(self, node: Node, level: int) -> np.ndarray:
        if level not in self.level_covariance_matrices:
            self.level_covariance_matrices[level] = self._compute_covariance_matrix_for_level(level)
        return self.level_covariance_matrices[level]

    def _compute_covariance_matrix_for_level(self, level: int) -> np.ndarray:
        level_nodes = self._get_level_nodes(self.tree.root, level, 0)
        n_assets = len(level_nodes)
        # Implement the logic to load the covariance matrix for the nodes at the given level
        # For demonstration purposes, let's create a random covariance matrix
        random_cov = np.random.rand(n_assets, n_assets)
        covariance_matrix = np.dot(random_cov, random_cov.T)
        return covariance_matrix

    def _get_level_nodes(self, node: Node, target_level: int, current_level: int) -> List[Node]:
        if current_level == target_level:
            return [node]
        level_nodes = []
        for child_node in node.children:
            level_nodes.extend(self._get_level_nodes(child_node, target_level, current_level + 1))
        return level_nodes


    def get_node_data(self, node: Node) -> Tuple[np.ndarray, np.ndarray]:
        # Load the covariance matrix using the node name
        cov_matrix = self.load_cov_matrix(node.name)

        # Create the expected returns array using the node's expected return
        expected_returns = np.full(cov_matrix.shape[0], node.expected_return)

        return expected_returns, cov_matrix

    def load_cov_matrix(self, node_name: str) -> np.ndarray:
        # Use the random_cov_matrix function to generate a covariance matrix for the node
        np.random.seed(hash(node_name) % 2**32)  # Seed the random number generator with the hash of the node name
        return self.random_cov_matrix()

    def random_cov_matrix(self, n_assets: int = 10, random_state: Optional[int] = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)

        # Generate a random correlation matrix
        random_corr = np.random.uniform(-1, 1, size=(n_assets, n_assets))
        random_corr = (random_corr + random_corr.T) / 2  # Make it symmetric
        random_corr[np.diag_indices_from(random_corr)] = 1  # Set diagonal to 1

        # Generate random standard deviations for each asset
        random_std_devs = np.random.uniform(0.05, 0.2, size=n_assets)

        # Compute the covariance matrix
        cov_matrix = np.outer(random_std_devs, random_std_devs) * random_corr

        return cov_matrix


class BaseOptimizer:
    def __init__(self, n_assets, tickers=None):
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.tickers = tickers
        self._risk_free_rate = None
        # Outputs
        self.weights = None

    def _make_output_weights(self, weights=None):
        if weights is None:
            weights = self.weights

        return collections.OrderedDict(zip(self.tickers, weights))

    def set_weights(self, input_weights):
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=5):
        if self.weights is None:
            raise AttributeError("Weights not yet computed")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)

    def save_weights_to_file(self, filename="weights.csv"):

        clean_weights = self.clean_weights()

        ext = filename.split(".")[1]
        if ext == "csv":
            pd.Series(clean_weights).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean_weights, fp)
        elif ext == "txt":
            with open(filename, "w") as f:
                f.write(str(dict(clean_weights)))
        else:
            raise NotImplementedError("Only supports .txt .json .csv")


class MeanVarianceOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self, expected_returns, covariance_matrix, risk_free_rate=0.0):
        self._risk_free_rate = risk_free_rate

        # Calculate the optimal weights using mean-variance optimization
        inverse_covariance = np.linalg.inv(covariance_matrix)
        ones = np.ones(self.n_assets)
        A = ones.T @ inverse_covariance @ ones
        B = ones.T @ inverse_covariance @ expected_returns
        C = expected_returns.T @ inverse_covariance @ expected_returns

        delta = A * C - B ** 2
        weights = (C * inverse_covariance @ ones - B * inverse_covariance @ expected_returns) / delta

        self.weights = weights
        return self.clean_weights()

    @staticmethod
    def from_node(node: Node, universe: Universe, level: int):
        level_nodes = universe._get_level_nodes(universe.tree.root, level, 0)
        n_assets = len(level_nodes)
        tickers = [level_node.name for level_node in level_nodes]
        expected_returns = np.array([level_node.expected_return for level_node in level_nodes])
        covariance_matrix = universe.loader(node, level)

        optimizer = MeanVarianceOptimizer(n_assets, tickers)
        optimizer.optimize(expected_returns, covariance_matrix)
        return optimizer


class Pipeline:
    def __init__(self, optimizers: List[Tuple[str, Callable]], universe: Universe):
        self.optimizers = optimizers
        self.universe = universe

    def run(self) -> None:
        self._run_optimizers(self.universe.tree.root, 0)

    def _run_optimizers(self, node: Node, level: int) -> List[float]:
        if not node.children:
            return [1.0]

        if level < len(self.optimizers):
            optimizer_name, optimizer_func = self.optimizers[level]
            parent_weights = optimizer_func(node, self.universe, level)
            child_weights = []

            for i, child_node in enumerate(node.children):
                child_node_weights = self._run_optimizers(child_node, level + 1)
                adjusted_weights = [w * parent_weights[i] for w in child_node_weights]
                child_weights.extend(adjusted_weights)

            return child_weights
        else:
            return [1.0]

    def run_and_show(self) -> None:
        self.run()
        self.universe.tree.draw()


# test

def dummy_optimizer(node: Node, universe: Universe, level: int) -> np.ndarray:
    # Implement a simple optimizer function that returns equal weights for each child node.
    return np.array([1 / len(node.children)] * len(node.children))

def mean_variance_optimizer(node: Node, universe: Universe, level: int) -> np.ndarray:
    # Retrieve the necessary data from the Universe for the optimizer
    expected_returns, cov_matrix = universe.get_node_data(node)

    # Create the MeanVarianceOptimizer instance
    optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)

    # Run the optimizer and return the optimized weights
    optimized_weights = optimizer.optimize()

    return optimized_weights

# Create a tree
tree = Tree("Root")
tree.insert("Root", "Stocks", expected_return=0.1, variance=0.05)
tree.insert("Stocks", "Korean_Stocks", expected_return=0.12, variance=0.06)
tree.insert("Stocks", "US_Stocks", expected_return=0.11, variance=0.055)
tree.insert("Root", "Bonds", expected_return=0.05, variance=0.03)
tree.insert("Bonds", "Government_Bonds", expected_return=0.04, variance=0.025)
tree.insert("Bonds", "Corporate_Bonds", expected_return=0.06, variance=0.035)

# Create a universe
universe = Universe(tree)

# Create a pipeline
pipeline = Pipeline(
    [("SAA", dummy_optimizer), ("TAA", mean_variance_optimizer)],
    universe
)

# Run the pipeline
asset_allocation = pipeline.run()

print("Asset allocation:")
for asset, weight in asset_allocation.items():
    print(f"{asset}: {weight}")
