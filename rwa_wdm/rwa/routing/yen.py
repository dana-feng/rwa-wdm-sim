"""Yen's algorithm, a.k.a. k-shortest paths algorithm as routing strategy

"""

from typing import List

import numpy as np
import networkx as nx
import time
from heapq import heappop, heappush


def yen(mat: np.ndarray, s: int, d: int, k: int) -> List[List[int]]:
    """Yen's routing algorithm, a.k.a. K-shortest paths

    Args:
        mat: Network's adjacency matrix graph
        s: source node index
        d: destination node index
        k: number of alternate paths

    Returns:
        :obj:`list` of :obj:`list`: a sequence of `k` paths

    """
    start_time = time.time()
    if s < 0 or d < 0:
        raise ValueError('Source nor destination nodes cannot be negative')
    elif s > mat.shape[0] or d > mat.shape[0]:
        raise ValueError('Source nor destination nodes should exceed '
                         'adjacency matrix dimensions')
    if k < 0:
        raise ValueError('Number of alternate paths should be positive')

    G = nx.from_numpy_array(mat, create_using=nx.Graph())
    paths = list(nx.shortest_simple_paths(G, s, d, weight=None))
    end_time = time.time() - start_time
    return paths[:k]




def yen_ksp_unweighted(graph, source, target, k=1):
    """
    Yen's algorithm for finding k-shortest loopless paths in an unweighted directed graph.

    Args:
        graph (nx.DiGraph): A directed graph.
        source (int): The source node.
        target (int): The target node.
        k (int): Number of shortest paths to find.

    Returns:
        List[List[int]]: A list of k shortest paths (as lists of nodes).
    """
    start_time = time.time()
    graph = nx.DiGraph(graph)
    # Initial shortest path from the source to the target
    first_path = nx.shortest_path(graph, source, target)
    if not first_path:
        return []
    
    shortest_paths = [first_path]
    potential_paths = []
    
    for i in range(1, k):
        for j in range(len(shortest_paths[-1]) - 1):
            spur_node = shortest_paths[-1][j]
            root_path = shortest_paths[-1][:j + 1]

            # Remove the links that are part of the previous shortest paths which share the same root path
            removed_edges = []
            for path in shortest_paths:
                if len(path) > j and root_path == path[:j + 1]:
                    edge = (path[j], path[j + 1])
                    if graph.has_edge(*edge):
                        removed_edges.append(edge)
                        graph.remove_edge(*edge)
            
            # Calculate the spur path from the spur node to the target
            try:
                spur_path = nx.shortest_path(graph, spur_node, target)
                total_path = root_path[:-1] + spur_path
                if total_path not in potential_paths:
                    heappush(potential_paths, (len(total_path), total_path))
            except nx.NetworkXNoPath:
                pass

            # Add back the edges that were removed
            for edge in removed_edges:
                graph.add_edge(*edge)
        
        if not potential_paths:
            break
        
        while potential_paths:
            _, path = heappop(potential_paths)
            if path not in shortest_paths:
                shortest_paths.append(path)
                break

    end_time = time.time() - start_time
    return shortest_paths

