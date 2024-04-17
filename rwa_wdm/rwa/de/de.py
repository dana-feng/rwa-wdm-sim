import logging
from typing import List, Tuple, Union
from ...net import Network
import random
import math
from ..routing import dijkstra, yen
from ..wlassignment import vertex_coloring, first_fit, random_fit
import copy

__all__ = (
    'DE',
)

logger = logging.getLogger(__name__)


class DE:
    def __init__(self, net: Network):
        self.M = 0.2  # Best control parameter for USA network https://link.springer.com/article/10.1007/s11107-013-0413-3
        self.RC = 0.5  # Control parameter
        self.NP = 100  # Control parameter
        self.Pop = []  # Population initialization
        self.k_shortest_paths = {}
        self.a2 = 0.5
        self.a1 = 1 - self.a2
        self.k = 3
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
                max_index = len(self.k_shortest_paths[n][m]) - 1
                APL += len(self.k_shortest_paths[n][m][max_index])
        APL = APL / (net.nnodes * (net.nnodes - 1))
        self.n1 = net.nnodes
        self.n2 = APL

    def initialize_population(self, net: Network):
        N = net.nnodes
        population = []
        for _ in range(self.NP):
            individual = []
            for n in range(N):
                for m in range(n + 1, N):
                    path = random.choice(self.k_shortest_paths[n][m])
                    index_of_path = self.k_shortest_paths[n][m].index(path)
                    individual.append(index_of_path)
            population.append(individual)
        self.Pop = population

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

    def mutation(self, selected_individuals):
        xr1, xr2, xr3 = selected_individuals
        result = [x - y for x, y in zip(xr2, xr3)]
        multiplier = self.M * random.random()
        intermed = [math.floor(x * multiplier) for x in result]  # Round the result to the nearest integer
        mi = [x + y for x, y in zip(intermed, xr1)]
        return mi

    def selection(self, ti, xi, net: Network):
        if ti is None:
            return xi
        if xi is None:
            return ti
        if self.objective_function(ti, net)[2] < self.objective_function(xi, net)[2]:
            return ti
        else:
            return xi

    def run(self, net: Network, k):
        self.initialize_population(net)
        max_generations = 100
        best_one = None
        while not max_generations <= 0:
            selected_population = []
            for xi in self.Pop:
                # print("Candidate", xi)
                best_one = self.selection(best_one, xi, net)
                selected_individuals = random.sample(self.Pop, 3)
                mutated_individual = self.mutation(selected_individuals)
                ti = []
                for mj, xj in zip(mutated_individual, xi):
                    rand = random.random()
                    if rand < self.RC:
                        element = mj
                    else:
                        element = xj
                    if element < 0 or element >= self.k:
                        element = random.randint(0, self.k - 1)                    
                    ti.append(element)
                selected_individual = self.selection(ti, xi, net)
                selected_population.append(selected_individual)
            self.Pop = selected_population
            max_generations -= 1
        for ind in selected_population:
            best_one = self.selection(best_one, ind, net)
        print("best one", best_one)
        return self.objective_function(best_one, net)
