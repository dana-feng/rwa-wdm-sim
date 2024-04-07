import logging
from typing import List, Tuple, Union
from ...net import Network
import random
import math
from ..routing import dijkstra, yen
from ..wlassignment import vertex_coloring, first_fit, random_fit
import copy


logger = logging.getLogger(__name__)


class SPFF:
    def __init__(self, net: Network):
        self.k = 3
        self.a2 = 0.5
        self.a1 = 1 - self.a2
        self.k_shortest_paths = {}
        for n in range(net.nnodes):
            # Initialize routing tables using the routing protocol
            self.k_shortest_paths[n] = {}
            for m in range(n + 1, net.nnodes):
                self.k_shortest_paths[n][m] = []
                self.k_shortest_paths[n][m].extend(yen(net.a, n, m, self.k))
            # calculate n1 and n2
        APL = 0
        for n in range(net.nnodes):
            for m in range(n + 1, net.nnodes):
                APL += len(self.k_shortest_paths[n][m][2])
        APL = APL / (net.nnodes * (net.nnodes - 1))
        self.n1 = net.nnodes
        self.n2 = APL


    def calculate_NWR(self, ind, net: Network):
        wavelength_count = {}
        max_wavelength = 0
        i = 0
        for n in range(net.nnodes):
            for m in range(n + 1, net.nnodes):
                index = int(ind[i])
                shortest_path_nodes = self.k_shortest_paths[n][m][index]
                for j in range(len(shortest_path_nodes) - 1):
                    segment_start = shortest_path_nodes[j]
                    segment_end = shortest_path_nodes[j + 1]
                    if segment_start not in wavelength_count:
                        wavelength_count[segment_start] = {}
                    if segment_end not in wavelength_count[segment_start]:
                        wavelength_count[segment_start][segment_end] = 0
                    wavelength_count[segment_start][segment_end] += 1
                    max_wavelength = max(max_wavelength, wavelength_count[segment_start][segment_end])
                i += 1
        return max_wavelength

    def objective_function(self, ind, net: Network):
        APL = 0
        i = 0
        for n in range(net.nnodes):
            for m in range(n + 1, net.nnodes):
                index = int(ind[i])
                APL += len(self.k_shortest_paths[n][m][index])
                i += 1
        APL = APL / ((net.nnodes * (net.nnodes - 1)) / 2)
        NWR = self.calculate_NWR(ind, net)
        return APL, NWR, self.a1 * (NWR / self.n1) + self.a2 * (APL / self.n2)


    def run(self, net: Network, k):
        total = (net.nnodes * (net.nnodes - 1)) // 2
        best_one = [0 for _ in range(total)]
        print("best one", best_one)
        return self.objective_function(best_one, net)
