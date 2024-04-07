from typing import Dict, List, Tuple
from collections import OrderedDict

from . import Network


class DE_Graph(Network):
    """Fish"""

    def __init__(self, ch_n):
        self._name = 'de_graph'
        self._fullname = u'DE Graph'
        self._s = 0
        self._d = 4
        super().__init__(ch_n,
                         len(self.get_nodes_2D_pos()),
                         len(self.get_edges()))
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get edges of the network"""
        return [
            (0, 2), (0, 1), (1, 0), (1, 3), (1,2), (2, 1), (2,0), (2,3), (2,4), (3, 2), (3, 1), (3, 4), (4, 2), (4, 3)
        ]

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        """Get position of the nodes on the bidimensional Cartesian plan"""
        return OrderedDict([
            ('0', (0,10)),
            ('1', (0, 0)),  
            ('2', (10, 10)),  
            ('3', (10, 0)),  
            ('4', (15, 5)), 
        ])