from typing import Dict, List, Tuple
from collections import OrderedDict

from . import Network


class Fish(Network):
    """Fish"""

    def __init__(self, ch_n):
        self._name = 'fish'
        self._fullname = u'Fish'
        self._s = 0
        self._d = 7
        super().__init__(ch_n,
                         len(self.get_nodes_2D_pos()),
                         len(self.get_edges()))
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get edges of the network"""
        return [
            (0, 3), (3,0), (1, 3), (3,1), (2, 3), (3, 2), (3, 4), (4, 3),
            (4, 7), (7, 4), (5, 7), (7, 5), (3, 6), (6, 3), (5,6), (6,5)     
        ]

    def get_nodes_2D_pos(self) -> Dict[str, Tuple[float, float]]:
        """Get position of the nodes on the bidimensional Cartesian plan"""
        return OrderedDict([
            ('0', (15, 2)),
            ('1', (10, 2)),  
            ('2', (5, 2)),  
            ('3', (10, 7)),  
            ('4', (16, 10)), 
            ('5', (7, 15)),  
            ('6', (5, 10)), 
            ('7', (10,15)), 
        ])