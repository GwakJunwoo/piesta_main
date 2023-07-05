from typing import List, Optional, Dict, Any, Tuple, Callable, Union
import numpy as np
import collections
import pandas as pd
import json

class Node:
    def __init__(self, name: str, **params: Any):
        self.name = name
        self.children: List[Node] = []
        self.params = params
        self.params['weight_bounds'] = params.get('weight_bounds', (0, 1))

    def add_child(self, child_node) -> None:
        self.children.append(child_node)
        
class Tree:
    def __init__(self, root_name: str):
        self.root = Node(root_name)
        #self.assumption = AssetAssumption

    def __repr__(self) -> str:
        return f'Tree({self.root.name})'

    def insert(self, parent_name: str, child_name: str, **params: Any) -> bool:
        parent_node = self._find_node(self.root, parent_name)
        if parent_node:
            child_node = Node(child_name, **params)
            parent_node.add_child(child_node)
            return True
        return False

    def draw(self) -> None:
        lines = self._build_tree_string(self.root, '')
        print('\n'.join(lines))

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
    
    def get_all_nodes(self) -> List[Node]:
        return self._collect_nodes(self.root, [])

    def _collect_nodes(self, node: Node, node_list: List[Node]) -> List[Node]:
        for child in node.children:
            node_list.append(child)
            self._collect_nodes(child, node_list)
        return node_list
    