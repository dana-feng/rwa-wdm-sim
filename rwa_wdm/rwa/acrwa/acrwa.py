import logging
from typing import List, Tuple, Union
from ...net import Network
import random
import math
from ..routing import dijkstra, yen
from ..wlassignment import vertex_coloring, first_fit, random_fit

__all__ = (
    'AntColonyRWA',
)

logger = logging.getLogger(__name__)


class AntColonyRWA(object):
    """
    Ant Colony Routing and Wavelength Assignment (ACRWA) algorithm.
    """

    def __init__(self) -> None:
        """Constructor"""
        # Initialize parameters
        self.alpha1 = 0.001  # Placeholder for alpha1 parameter
        self.rho1 = 0.001  # Placeholder for rho1 parameter
        self.beta = 1 # Placeholder for beta parameter
        self.omega = 0.75  # Placeholder for omega parameter
        self.phi = 0.75  # Placeholder for phi parameter
        self.r0 = 0.9 # Placeholder for r0 parameter
        self.candidate_nodes_list = {}
        self.candidate_lambdas_list = {}
        self.routing_tables = {} # self.routing_tables[n][m] holds the k shortest paths from n to m; since we have only one port per node, this means this will only hold one path
        self.desirability = {}
        self.pheromone_table = {}
    
    def initialization(self, net: Network):
        """
        Initialize parameters, routing tables, candidate lists, and desirability values.
        """
        # Initialization loop for each node
        for n in range(net.nnodes):
            # Initialize routing tables using the routing protocol
            self.routing_tables[n] = {}
            for m in range(net.nnodes):
                shortest_paths = yen(net.a, n, m, 11)
                self.routing_tables[n][m] = shortest_paths# Routing table will be a list of the k shortest paths
                # Build candidate nodes list and initialize desirability values
                self.candidate_nodes_list[n] = list(set(path[1] for path in self.routing_tables[n][m] if len(path) > 1)) # Assuming self.routing_tables[n][m] returns a list of shortest paths
                self.desirability[n] = 1/len(self.routing_tables[n][m][0])
                # Initialize candidate lambdas list
                self.candidate_lambdas_list[n] = list(range(net.nchannels))
        # Initialize the pheromone table
        for i in range(net.nnodes):
            for j in range(net.nnodes):
                # Use range function directly, no need to convert to list

                for wavelength in range(net.nchannels):
                    # Use range function directly, no need to convert to list

                    path_length = len(self.routing_tables[i][j][0])
                    congestion_level = net.a[i][j]  # Assuming this is the congestion level, not the number of free wavelengths

                    if i not in self.pheromone_table:
                        self.pheromone_table[i] = {}  # Initialize if not exists

                    if j not in self.pheromone_table[i]:
                        self.pheromone_table[i][j] = {}  # Initialize if not exists

                    if wavelength not in self.pheromone_table[i][j]:
                        self.pheromone_table[i][j][wavelength] = 0  # Initialize if not exists

                    self.pheromone_table[i][j][wavelength] = 1 / (path_length * net.nnodes) # TODO not sure if this is right
    
    def RWA(self, nodei, net):
        best_product = None
        u = None
        lmbda = None
        # Initialize routing tables using the routing protocol
        for nodej in self.candidate_nodes_list[nodei]:
            for wavelengthk in self.candidate_lambdas_list[nodei]:
                if net.n[nodei][nodej][wavelengthk]: # if free
                    curr_product = self.pheromone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
                    if best_product is None or best_product < curr_product:
                        best_product = curr_product
                        u = nodej
                        lmbda = wavelengthk
        return (net.s, u, lmbda)
    
    def exploit(self, nodei, wavelengthk):
        best_product = None
        u = None
        # Initialize routing tables using the routing protocol
        for nodej in self.candidate_nodes_list[nodei]:
            if nodej == 6 or nodej == 4:
                print("nodej", nodej, "pheremone", self.pheromone_table[nodei][nodej][wavelengthk])
            curr_product = self.pheromone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
            if best_product is None or best_product < curr_product:
                best_product = curr_product
                u = nodej
        if (u == 4 and nodei == 3) or (u ==6 and nodei == 3):
            print("exploit to go to ", u, "wavelength", wavelengthk)
        return u

    def explore(self, nodei, wavelengthk):
        empirical_distribution = {}
        running_sum = 0
        for nodej in self.candidate_nodes_list[nodei]:
            curr_product = self.pheromone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
            running_sum += curr_product
            empirical_distribution[nodej] = curr_product
        
        empirical_distribution = {key: value / running_sum for key, value in empirical_distribution.items()}

        values = list(empirical_distribution.keys())
        probabilities = list(empirical_distribution.values())
        u = random.choices(values, weights=probabilities)[0]
        if (u == 4 and nodei == 3) or (u ==6 and nodei == 3):
            print("exploit to go to ", u, "wavelength", wavelengthk)
        return u


    def run(self, net: Network, k:int) -> Tuple[List[int], Union[int, None]]:
        """Run the ACRWA algorithm

        Args:
            net: network object representing the network.

        Returns:
            A tuple containing the route as a list of router indices and wavelength index upon RWA success.
        """
        self.success = False

        # Placeholder variables for route and wavelength
        route = []  # Placeholder for route
        wavelength = None  # Placeholder for wavelength

        self.xm = [] # Placeholder for xm(t)
        self.Antblocked = False
        self.currSrc = net.s
        self.u = None
        self.lmbda = None

        # Initial RWA if xm(t) is empty
        if not self.xm:
            # initial RWA using Eq. (3)
            self.xm = [self.RWA(net.s, net)]
            self.currSrc, self.u, self.lmbda = self.xm[0]
            wavelength = self.lmbda
            if wavelength is None:
                self.Antblocked = True
        # Main loop until ant arrives destination or self.Antblocked is true
        while not (self.ant_arrives_destination(net) or self.Antblocked):
            # Placeholder for checking if there's any j in Nmn(t)
            if self.candidate_nodes_list[self.currSrc]:
                r = random.random()
                if r <= self.r0:
                    # Explits the previous pheromone deposits Eq. (4)
                    self.u = self.exploit(self.currSrc, self.lmbda)
                else:
                    # Look for new route Eq. (5)
                    self.u = self.explore(self.currSrc, self.lmbda)
                self.xm.append((self.currSrc, self.u, self.lmbda))
                if not self.ant_reservation_on_link(net):
                    self.Antblocked = True
                if not self.Antblocked:
                    # Run positive local updating rule using Eq. (9)
                    self.run_positive_local_updating_rule(net)
                self.currSrc = self.u
            else:
                self.Antblocked = True
        route = [node[0] for node in self.xm[1:]] + [net.d]
        # Once done, the reverse ant runs
        # Repeat until xm(t) is empty or reverse ant arrives origin node
        self.reverse_path_length = 0
        while self.xm and not self.reverse_ant_arrives_origin_node(net):
            link = self.xm.pop()
            src, dest, k = link
            # Run global updating rule using Eq. (6)
            self.run_global_updating_rule(link, net)
            self.currSrc = dest
            self.reverse_path_length += 1
        # wavelength = random_fit(net, route)
        if wavelength is not None:
            print("route", route, "wavelength", wavelength)
        return route, wavelength

    # Placeholder functions
    def ant_arrives_destination(self, net):
        self.success = True
        return self.currSrc == net.d

    def ant_reservation_on_link(self, net):
        return net.n[self.currSrc][self.u][self.lmbda]

    def reverse_ant_arrives_origin_node(self, net):
        return self.currSrc == net.s
    
    def run_positive_local_updating_rule(self, net):
        ant_path_length = len(self.xm)
        delta = ant_path_length - len(self.routing_tables[self.currSrc][net.s][0])
        tauijk = self.pheromone_table[self.currSrc][net.d][self.lmbda]
        self.pheromone_table[self.currSrc][net.d][self.lmbda] = tauijk + self.alpha1*math.exp(-1*self.phi*delta)
    
    def get_gammaij(self, link):
        if link in self.xm:
            if self.success:
                return 1
            else:
                return -1
        else:
            return 0
        
    def run_global_updating_rule(self, link, net):
        src, dest, k = link
        tauijk = self.pheromone_table[src][dest][k]
        gammaij = self.get_gammaij(link)
        delta = self.reverse_path_length - len(self.routing_tables[self.currSrc][net.s][0])
        delta_tau_ijk = math.exp(-1*self.omega*delta)
        self.pheromone_table[src][dest][k] = (1-self.rho1)*tauijk + self.rho1*gammaij*delta_tau_ijk