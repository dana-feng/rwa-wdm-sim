import logging
from typing import List, Tuple, Union
from ...net import Network
import random
import math
from ..routing import dijkstra, yen

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
        self.alpha1 = 0  # Placeholder for alpha1 parameter
        self.rho1 = 0  # Placeholder for rho1 parameter
        self.beta = 1 # Placeholder for beta parameter
        self.omega = 0  # Placeholder for omega parameter
        self.phi = 0  # Placeholder for phi parameter
        self.r0 = 0  # Placeholder for r0 parameter
        self.candidate_nodes_list = {}
        self.candidate_lambdas_list = {}
        self.routing_tables = {} # self.routing_tables[n][m] holds the k shortest paths from n to m; since we have only one port per node, this means this will only hold one path
        self.desirability = {}
        self.phereomone_table = {}
        self.r0 = 0.5
    
    def initialization(self, net: Network):
        """
        Initialize parameters, routing tables, candidate lists, and desirability values.
        """
        # Assuming self.net.nodes gives the list of nodes
        self.net = net

        # Initialization loop for each node
        for n in list(range(self.nnodes)):
            # Initialize routing tables using the routing protocol
            self.routing_tables[n] = [yen(self.net.a, n, m, 1) for m in range(self.nnodes)]
            for m in list(range(self.nnodes)):
                if n == m:
                    pass
                # Build candidate nodes list and initialize desirability values
                self.candidate_nodes_list[n] = [path[1] for path in self.routing_tables[n][m]] # assuming self.routing_tables[n][m] returns a list of shortest paths
                self.desirability[n] = 1/len(self.routing_tables[n][m][0])
                # Initialize candidate lambdas list
                self.candidate_lambdas_list[n] = list(range(self.net.nchannels))
        # Intitialize the pheremone table
        self.
    
    
    def RWA(self, nodei):
        best_product = None
        u = None
        lmbda = None
        # Initialize routing tables using the routing protocol
        for nodej in self.candidate_nodes_list[nodei]:
            for wavelengthk in range(self.candidate_lambdas_list[nodei]):
                curr_product = self.phereomone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
                if best_product is None or best_product < curr_product:
                    best_product = curr_product
                    u = nodej
                    lmbda = wavelengthk
        return [(self.net.s, u, lmbda)]
    
    def exploit(self, nodei, wavelengthk):
        best_product = None
        u = None
        # Initialize routing tables using the routing protocol
        for nodej in self.candidate_nodes_list[nodei]:
            curr_product = self.phereomone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
            if best_product is None or best_product < curr_product:
                best_product = curr_product
                u = nodej
        return u

    def explore(self, nodei, wavelengthk):
        empirical_distribution = {}
        running_sum = 0
        for nodej in self.candidate_nodes_list[nodei]:
            curr_product = self.phereomone_table[nodei][nodej][wavelengthk]*(self.desirability[nodei]**self.beta)
            running_sum += curr_product
            empirical_distribution[nodej] = curr_product
        
        empirical_distribution = {key: value / running_sum for key, value in empirical_distribution.items()}

        values = list(empirical_distribution.keys())
        probabilities = list(empirical_distribution.values())
        u = random.choices(values, weights=probabilities)[0]
        return u


    def run(self, net: Network) -> Tuple[List[int], Union[int, None]]:
        """Run the ACRWA algorithm

        Args:
            self.net: self.network object representing the self.network.

        Returns:
            A tuple containing the route as a list of router indices and wavelength index upon RWA success.
        """
        self.success = False

        # Placeholder variables for route and wavelength
        route = []  # Placeholder for route
        wavelength = None  # Placeholder for wavelength

        self.xm = [] # Placeholder for xm(t)
        Antblocked = False
        self.currSrc = self.net.s
        self.u = None
        self.lmbda = None

        # Initial RWA if xm(t) is empty
        if not self.xm:
            # initial RWA using Eq. (3)
            self.xm = self.RWA(self.net)
            self.currSrc, self.u, self.lmbda = self.xm[0]
            wavelength = self.lmbda

        # Main loop until ant arrives destination or Antblocked is true
        while not (self.ant_arrives_destination() or Antblocked):
            # Placeholder for checking if there's any j in Nmn(t)
            if self.candidate_nodes_list[self.currSrc]:
                r = random.randint()
                if r <= self.r0:
                    # Explits the previous pheromone deposits Eq. (4)
                    self.u = self.exploit(self.currSrc, self.xm[1])
                else:
                    # Look for new route Eq. (5)
                    self.u = self.explore(self.currSrc, self.xm[1])
                self.xm.append((self.currSrc, self.u, self.lmbda))
                if not self.ant_reservation_on_link():
                    Antblocked = True
                if not Antblocked:
                    # Run positive local updating rule using Eq. (9)
                    self.run_positive_local_updating_rule()
            else:
                Antblocked = True
        route = [node[0] for node in self.x]
        # Once done, the reverse ant runs
        # Repeat until xm(t) is empty or reverse ant arrives origin node
        while self.xm and not self.reverse_ant_arrives_origin_node():
            link = self.xm.pop()
            src, dest, k = link
            # Run global updating rule using Eq. (6)
            self.run_global_updating_rule(link)
            self.currSrc = dest
            self.reverse_path_length += 1

        return route, wavelength

    # Placeholder functions
    def ant_arrives_destination(self):
        self.success = True
        return self.currSrc == self.net.d

    def ant_reservation_on_link(self):
        return self.n[self.currSrc][self.u][self.lmbda]

    def reverse_ant_arrives_origin_node(self):
        return self.currSrc == self.net.src
    
    def run_positive_local_updating_rule(self):
        delta = self.reverse_path_length - len(self.routing_tables[self.currSrc][self.net.s][0])
        tauijk = self.phereomone_table[self.currSrc][self.net.d][self.xm[2]]
        self.phereomone_table[self.currSrc][self.net.d][self.xm[2]] = tauijk + self.alpha1*math.exp(-1*self.phi*delta)
    
    def get_gammaij(self, link):
        if link in self.xm:
            if self.success:
                return 1
            else:
                return -1
        else:
            return 0
        
    def run_global_updating_rule(self, link):
        src, dest, k = link
        tauijk = self.phereomone_table[src][dest][k]
        gammaij = self.get_gammaij(link)
        delta = len(self.reverse_path) - len(self.routing_tables[self.currSrc][self.net.s][0])
        delta_tau_ijk = math.exp(-1*self.omega*delta)
        self.phereomone_table[src][dest][k] = (1-self.rho1)*tauijk + self.rho1*gammaij*delta_tau_ijk