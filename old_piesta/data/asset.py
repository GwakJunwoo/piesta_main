from collections import defaultdict
from functools import reduce
import operator
from typing import Dict, List, Optional
from graphviz import Digraph
from treelib import Tree, Node

class Universe:
    def __init__(self, universe: Optional[Dict] = None):
        self._update(universe)

    def remove(self, name):
        self._remove(name, self._universe)

    def get_universe(self) -> Dict:
        return self._universe

    def get_universe_by_layer(self) -> Dict:
        return dict(self._hierarchy)

    def get_keys_layer(self) -> List:
        return self._hierarchy_list

    def get_last_layer(self) -> List:
        return self._last_assets

    def diagram(self, filename=None, save_png=False):
        g = Digraph('G', filename=filename or 'universe_diagram', format='png')
        for asset_type in self._universe:
            with g.subgraph(name=f'cluster_{asset_type}') as cluster:
                cluster.attr(label=asset_type)
                for asset_region in self._universe[asset_type]:
                    for ticker in self._universe[asset_type][asset_region]:
                        cluster.node(ticker, label=f"{ticker}\n({asset_region})")
        if save_png:
            g.render(filename=filename or 'universe_diagram', view=False)
        else:
            g.view()

    def _remove(self, name, d):
        if self._universe is not None:
            self._recur_remove_func(name, d)
            self._update(self._universe)

    def _update(self, universe: Optional[Dict] = None):
        self._universe = universe or self._generate_sample()
        self._depth = self._get_depth(self._universe)
        self._hierarchy_list = self._get_keys_by_layer(self._universe)
        self._last_assets = self._get_bottom_values_by_layer(self._universe)
        self._hierarchy = defaultdict(Optional[Dict or List])

        cnt = 0
        for level in range(self._depth - 1):
            self._hierarchy[f'L{cnt}'] = self._hierarchy_list[level]
            cnt += 1
        self._hierarchy[f'L{cnt + 1}'] = self._last_assets

    def _recur_remove_func(self, name, d):
        if isinstance(d, dict):
            for key, value in list(d.items()):
                if key == name:
                    del d[key]
                else:
                    self._remove(name, value)
        elif isinstance(d, list):
            for item in d:
                self._remove(name, item)
        elif isinstance(d, str):
            del d

    def _get_depth(self, d: Dict) -> int:
        if isinstance(d, dict):
            return 1 + (max(map(self._get_depth, d.values())) if d else 0)
        return 1

    def _get_keys_by_layer(self, d):
        result = [[]]
        for key, value in d.items():
            if isinstance(value, dict):
                sub_result = self._get_keys_by_layer(value)
                for i in range(len(sub_result)):
                    if i >= len(result) - 1:
                        result.append([])
                    result[i + 1].extend(sub_result[i])
            result[0].append(key)
        return result

    def _get_bottom_values(self, d):
        if isinstance(d, dict):
            return [value for val in d.values() for value in self._get_bottom_values(val)]
        else:
            return [d]


    def _get_bottom_values_by_layer(self, d):
        return list(reduce(operator.add, self._get_bottom_values(d) if d else []))

    def _generate_sample(self) -> Dict:
        _universe = {
            'Stock': {
                'Korea': ['Ticker_0', 'Ticker_1'],
                'US': ['Ticker_2', 'Ticker_3'],
                'Europe': ['Ticker_4', 'Ticker_5'],
                'Japan': ['Ticker_6', 'Ticker_7'],
                'China': ['Ticker_8', 'Ticker_9']
            },
            'Bond': {
                'Developed': ['Ticker_10', 'Ticker_11'],
                'Emerging': ['Ticker_12', 'Ticker_13']
            },
            'Alternative': {
                'Real estate': ['Ticker_14', 'Ticker_15'],
                'Hedge fund': ['Ticker_16', 'Ticker_17']
            },
            'Commodity': {
                'Metal' : [],
                'Grains': ['Ticker_20', 'Ticker_21'],
                'Energy': ['Ticker_22', 'Ticker_23']
            },
            'Currency': {
                'USDKRW': ['Ticker_24', 'Ticker_25'],
                'USDJPY': ['Ticker_26', 'Ticker_27'],
                'USDEUR': ['Ticker_28', 'Ticker_29']
            }
        }
        return _universe


class UniverseTree(Universe):
    def __init__(self, universe: Optional[Dict] = None):
        super().__init__(universe)
        self._build_tree()

    def _build_tree(self):
        self.tree = Tree()
        self.tree.create_node("Universe", "root")

        def add_nodes(parent_key, data):
            for key, value in data.items():
                node_id = f"{parent_key}-{key}"
                self.tree.create_node(key, node_id, parent=parent_key)
                if isinstance(value, dict):
                    add_nodes(node_id, value)

        add_nodes("root", self._universe)

    def remove(self, name):
        nodes_to_remove = self.tree.search_nodes(name=name)
        for node in nodes_to_remove:
            self.tree.remove_node(node.identifier)

    def _update(self, universe: Optional[Dict] = None):
        super()._update(universe)
        self._build_tree()

    def diagram(self, filename=None, save_png=False):
        g = Digraph('G', filename=filename or 'universe_tree_diagram', format='png')
        
        def create_nodes(node: Node):
            for child in self.tree.children(node.identifier):
                g.node(child.identifier, label=child.tag)
                g.edge(node.identifier, child.identifier)
                create_nodes(child)

        create_nodes(self.tree.get_node("root"))

        if save_png:
            g.render(filename=filename or 'universe_tree_diagram', view=False)
        else:
            g.view()