import random
from typing import Dict, List, Tuple
from collections import OrderedDict

from . import Network

class DE_Graph_Custom(Network):
    """Custom DE"""

    def __init__(self, ch_n, n):
        self._name = 'de_graph'
        self._fullname = u'DE Graph'
        self._s = 0
        self._d = 4
        self.num_nodes = n
        super().__init__(ch_n,
                         len(self.get_nodes_2D_pos()),
                         len(self.get_edges()))

    def get_edges(self) -> List[Tuple[int, int]]:
        """Get edges of the network"""
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and random.random() < 0.5:  # Adjust the probability of edge creation as needed
                    edges.append((i, j))
        return edges

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        """Get position of the nodes on the bidimensional Cartesian plan"""
        node_positions = OrderedDict()
        for i in range(self.num_nodes):
            node_positions[str(i)] = (random.uniform(0, 20), random.uniform(0, 20))  # Adjust range as needed
        return node_positions
