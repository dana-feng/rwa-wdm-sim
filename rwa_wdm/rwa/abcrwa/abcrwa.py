import logging
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from ..routing import yen
from ...net import Network

__all__ = ('ArtificialBeeColonyRWA',)

logger = logging.getLogger(__name__)


class ArtificialBeeColonyRWA:
    """
    Artificial Bee Colony algorithm for Routing and Wavelength Assignment (ABCRWA).

    This class implements the ABC algorithm to find optimal routing and wavelength
    assignments in optical networks under given constraints.

    Attributes:
        colony_size (int): Number of solutions (food sources) in the colony.
        max_cycles (int): Maximum number of cycles to run the optimization.
    """

    def __init__(self, colony_size: int = 20, max_cycles: int = 20) -> None:
        """
        Initializes the ABCRWA instance with specified colony size and number of cycles.

        Args:
            colony_size (int): Number of bees/solutions in the colony.
            max_cycles (int): Maximum number of iterations for the algorithm.
        """
        self.colony_size = colony_size
        self.max_cycles = max_cycles
        self.routing_table = None

    def generate_routing_table(self, net: Network):
        """
        Initializes the routing table with the k-shortest paths between all pairs of nodes
        in the network using the Yen's algorithm.

        Args:
            net (Network): The network topology on which the RWA is being performed.

        Returns:
            routing_table: Dict[(int, int), List[List[int]]], a dictionary of tuples (i,j)
            that corresponds to the list of k-shortest paths from node i to node j in the network.
        """
        routing_table = {}
        # Loop over all pairs of nodes to populate the routing table
        for i in range(net.nnodes):
            for j in range(net.nnodes):
                if i != j:
                    shortest_paths = yen(net.a, i, j, self.colony_size)
                    routing_table[(i, j)] = shortest_paths
        return routing_table

    def initialization(self, net: Network) -> None:
        """
        Initializes the population of solutions and evaluates them.

        Args:
            net (Network): The network topology on which the RWA is being performed.
        """
        if not self.routing_table:
            self.routing_table = self.generate_routing_table(net)

        self.solutions = self.generate_initial_solutions(net)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluate_population(net)

    def generate_initial_solutions(self, net: Network) -> List[Dict[str, List[int]]]:
        """
        Generates initial solutions based on shortest paths for the colony.

        Args:
            net (Network): The network topology to generate paths for.

        Returns:
            List[Dict[str, List[int]]]: A list of initial solutions containing paths and wavelengths.
        """
        initial_solutions = []
        shortest_paths = self.routing_table[(net.s, net.d)]
        for path in shortest_paths:
            wavelength = random.randint(0, net.nchannels - 1)
            initial_solutions.append({'path': path, 'wavelength': wavelength})
        return initial_solutions

    def generate_random_solution(self, net):
        # Generate a random route and wavelength TODO: RECONSIDER -- curently possibly invalid solution
        path = [net.s]
        current = net.s
        while current != net.d:
            neighbors = [j for j in range(
                net.nnodes) if net.a[current][j] == 1]
            current = random.choice(neighbors)
            path.append(current)
        wavelength = random.randint(0, net.nchannels - 1)
        return {'path': path, 'wavelength': wavelength}

    def evaluate_population(self, net: Network) -> None:
        """
        Evaluates all solutions in the population to find the best solution.

        Args:
            net (Network): The network topology to evaluate each solution.
        """
        for solution in self.solutions:
            fitness = self.fitness_function(solution, net)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

    def fitness_function(self, solution: Dict[str, any], net: Network) -> float:
        """
        Calculates the fitness of a solution based on path length and transmission success.

        Args:
            solution (Dict[str, any]): A solution containing a path and a wavelength.
            net (Network): The network topology for checking path validity.

        Returns:
            float: The fitness value of the solution, with lower being better.
        """
        path, wavelength = solution['path'], solution['wavelength']
        if any(not net.n[path[i-1]][path[i]][wavelength] for i in range(1, len(path))):
            return float('inf')  # Collision occurred
        return len(path)  # The shorter the path, the better

    def employed_bee_phase(self, net: Network) -> None:
        """
        Modifies each solution in the population slightly, mimicking the employed bee behavior.

        Args:
            net (Network): The network topology used for modification context.
        """
        for i in range(self.colony_size):
            solution = self.solutions[i]
            self.modify_solution(solution, net)

    def modify_solution(self, solution: Dict[str, any], net: Network) -> None:
        """
        Modifies a solution by either changing one node in the path or the wavelength.

        Args:
            solution (Dict[str, any]): The solution to be modified.
            net (Network): The network topology for context.

        """
        path = solution['path'][:]
        if len(path) > 2 and random.random() < 0.5:
            pos = random.randint(1, len(path) - 2)
            neighbors = [j for j in range(
                net.nnodes) if net.a[path[pos-1]][j] == 1 and j != path[pos+1]]
            if neighbors:
                path[pos] = random.choice(neighbors)
            solution['path'] = path
        else:
            new_wavelength = random.randint(0, net.nchannels - 1)
            solution['wavelength'] = new_wavelength

        new_fitness = self.fitness_function(solution, net)
        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.best_solution = solution.copy()

    # GREEDY MODFICATION
    def greedy_modify_solution(self, solution: Dict[str, any], net: Network) -> None:
        """
        Tries to modify a solution by changing one node in the path or the wavelength.
        The modification is kept only if it improves the fitness of the solution.

        Args:
            solution (Dict[str, any]): The solution to be modified.
            net (Network): The network topology for context.
        """
        # Store original details to possibly revert changes
        original_path = solution['path'][:]
        original_wavelength = solution['wavelength']
        original_fitness = self.fitness_function(solution, net)

        # Decide whether to change the path or the wavelength
        if len(original_path) > 2 and random.random() < 0.5:
            pos = random.randint(1, len(original_path) - 2)
            neighbors = [j for j in range(
                net.nnodes) if net.a[original_path[pos-1]][j] == 1 and j != original_path[pos+1]]
            if neighbors:
                solution['path'][pos] = random.choice(neighbors)
        else:
            solution['wavelength'] = random.randint(0, net.nchannels - 1)

        # Calculate new fitness and decide whether to keep the changes
        new_fitness = self.fitness_function(solution, net)
        if new_fitness >= original_fitness:
            # Revert to original if no improvement
            solution['path'] = original_path
            solution['wavelength'] = original_wavelength
        else:
            # Update best solution if this is the best seen so far globally
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = solution.copy()

    def onlooker_bee_phase(self, net: Network) -> None:
        """
        Onlooker bees choose and further modify solutions based on their fitness probabilities.

        Args:
            net (Network): The network topology used for evaluating solutions.
        """
        fitnesses = [1 / (self.fitness_function(sol, net) if self.fitness_function(
            sol, net) != float('inf') else 1) for sol in self.solutions]
        total_fitness = sum(fitnesses)
        probs = [f / total_fitness for f in fitnesses]

        for _ in range(self.colony_size):
            index = random.choices(
                range(self.colony_size), weights=probs, k=1)[0]
            self.modify_solution(
                self.solutions[index], net)

    def scout_bee_phase(self, net: Network) -> None:
        """
        Scout bees randomly generate new solutions to replace poor ones.
        TODO: This could be modified so that solutions are abandoned every x iterations
        that the solution does not improve.

        Args:
            net (Network): The network topology used for generating new solutions.
        """
        for i in range(self.colony_size):
            if random.random() < 0.2:  # Low probability to regenerate a solution
                self.solutions[i] = self.generate_random_solution(
                    net)  # TODO: Reconsider, as the solution might no longer be valid
                fitness = self.fitness_function(self.solutions[i], net)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.solutions[i].copy()

    def run(self, net: Network) -> Tuple[Optional[List[int]], Optional[int]]:
        """
        Runs the main optimization loop of the ABCRWA algorithm.

        Args:
            net (Network): The network topology to optimize routing and wavelength assignment.

        Returns:
            Tuple[Optional[List[int]], Optional[int]]: The best route and wavelength if found, otherwise None.
        """
        self.initialization(net)
        for _ in range(self.max_cycles):
            self.employed_bee_phase(net)
            self.onlooker_bee_phase(net)
            self.scout_bee_phase(net)

        if self.best_solution:
            return self.best_solution["path"], self.best_solution["wavelength"]
        return None, None
