from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
import collections
import pandas as pd
import json


class Node:
    def __init__(self, name: str, **params: Any):
        self.name = name
        self.children: List[Node] = []
        self.expected_return = params.get('expected_return', None)
        self.variance = params.get('variance', None)
        self.covariance_matrix = params.get('covariance_matrix', None)

    def add_child(self, child_node: 'Node') -> None:
        self.children.append(child_node)


class Tree:
    def __init__(self, root_name: str):
        self.root = Node(root_name)

    def __repr__(self) -> str:
        return f'Tree({self.root.name})'

    def insert(self, parent_name: str, child_name: str) -> bool:
        parent_node = self._find_node(self.root, parent_name)
        if parent_node:
            parent_node.add_child(Node(child_name))
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

    def insert(self, parent_name: str, child_name: str, **params: Any) -> bool:
        return self.tree.insert(parent_name, child_name, **params)


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
    def from_node(node: Node):
        n_assets = len(node.children)
        tickers = [child.name for child in node.children]
        expected_returns = np.array([child.expected_return for child in node.children])
        covariance_matrix = np.array([child.covariance_matrix for child in node.children])

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
            parent_weights = optimizer_func(node)
            child_weights = []

            for child_node in node.children:
                child_node_weights = self._run_optimizers(child_node, level + 1)
                adjusted_weights = [w * parent_weights for w in child_node_weights]
                child_weights.extend(adjusted_weights)

            return child_weights
        else:
            return [1.0]

    def run_and_show(self) -> None:
        self.run()
        self.universe.tree.draw()
