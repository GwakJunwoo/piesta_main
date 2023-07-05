from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from data.Tree import *

class pipeline:
    def __init__(self, optimizers: List[Tuple[str, Callable]], universe: Tree):
        """
        pipeline 클래스는 재귀적인 함수호출로 계층적 자산배분을 수행한다.

        Attributes:
        - run(self): 재귀함수를 호출하여 각 자산군별 투자비중을 반환한다.
        - _optimize_node(): 큰 문제를 작은문제로 나누어 작동한다.
        - _update_node(self, assumption_dict:Dict, dates:str): 자산 가정데이터를 전달받아 dates를 기준으로 각 node의 값을 업데이트한다.
        - _get_nodes_bounds(self, nodes: List[Node]): List[Node]를 전달받아 각 설정된 weight_bounds를 List로 반환한다.
        """

        self.optimizers = optimizers
        self.universe = universe
        self.covariance = None

    def run(self) -> Dict[str, float]:
        allocations = {}
        root_node = self.universe.root
        self._optimize_node(root_node, 1, allocations)
        return allocations

    def _optimize_node(self, node: Node, depth: int, allocations: Dict[str, float], parent_weight: float = 1.0) -> None:
        if depth == 1 and node.children:
            optimizer_name, optimizer_func = self.optimizers[0]
            node_weights = optimizer_func(node.children, weight_bounds = self._get_nodes_bounds(node.children))
            for child_node, weight in zip(node.children, node_weights):
                allocations[child_node.name] = weight
                self._optimize_node(child_node, depth + 1, allocations, weight)
        elif node.children:
            optimizer_name, optimizer_func = self.optimizers[depth - 1]
            node_weights = optimizer_func(node.children, weight_bounds = self._get_nodes_bounds(node.children))
            for child_node, weight in zip(node.children, node_weights):
                allocations[child_node.name] = weight * parent_weight
                self._optimize_node(child_node, depth + 1, allocations, weight * parent_weight)
        return

    def _update_node(self, assumption_dict:Dict, dates:str):
        # 이때 assumption_dict은 returns, covariance 등을 담고있으며 전기간에 대한, 전자산군에 대한 값을 담고있다.
        # 데이터 시작일은 그대로 이지만, 연산 시점에 따라서 end_date이 다르다.
        # 이를 series_at_time 이라는 이름으로 전달받은 dates를 기준으로 각 노드의 params를 업데이트 해준다.
        # 하지만 해당 날짜가 비영업일이거나 연산상의 이유로 누락된 경우 오류가 발생한다.
        nodes_list = self.universe.get_all_nodes()
        for key, df in assumption_dict.items():
            series_at_time = df.loc[dates,:]
            for node in nodes_list:
                node.params[key] = series_at_time[node.name]
                
    def _get_nodes_bounds(self, nodes: List[Node]) -> List[Tuple]:
        bounds = [node.params['weight_bounds'] for node in nodes]
        return bounds