"""RWA simulator main function

"""

# [1] https://la.mathworks.com/matlabcentral/fileexchange/4797-wdm-network-blocking-computation-toolbox

import logging
from timeit import default_timer  # https://stackoverflow.com/a/25823885/3798300
from typing import Callable
from argparse import Namespace

import numpy as np

from .io import write_bp_to_disk, write_it_to_disk, plot_bp
from .net import Network, DE_Graph_Custom
from .rwa import spff_algorithm, de_algorithm
__all__ = (
    'get_net_instance_from_args',
    'get_rwa_algorithm_from_args',
    'simulator'
)

logger = logging.getLogger(__name__)
global upper_bound


def get_net_instance_from_args(topname: str, numch: int) -> Network:
    """Instantiates a Network object from CLI string identifiers

    This is useful because rwa_wdm supports multiple network topology
    implementations, so this function acts like the instance is created
    directly.

    Args:
        topname: short identifier for the network topology
        numch: number of wavelength channels per network link

    Returns:
        Network: network topology instance

    Raises:
        ValueError: if `topname` is not a valid network identifier

    """
    global upper_bound
    upper_bound = 12
    if topname == 'nsf':
        from .net import NationalScienceFoundation
        return NationalScienceFoundation(numch)

    elif topname == 'clara':
        from .net import CooperacionLatinoAmericana
        return CooperacionLatinoAmericana(numch)
    elif topname == 'janet':
        from .net import JointAcademicNetwork
        return JointAcademicNetwork(numch)
    elif topname == 'rnp':
        from .net import RedeNacionalPesquisa
        return RedeNacionalPesquisa(numch)
    elif topname == 'fish':
        upper_bound = 7
        from .net import Fish
        return Fish(numch)
    elif topname == "de_graph":
        from .net import DE_Graph
        return DE_Graph(numch)
    else:
        raise ValueError('No network named "%s"' % topname)


def get_rwa_algorithm_from_args(r_alg: str, wa_alg: str, rwa_alg: str,
                                ga_popsize: int, ga_ngen: int,
                                ga_xrate: float, ga_mrate: float, net: Network) -> Callable:
    """Defines the main function to perform RWA from CLI string args

    Args:
        r_alg: identifier for a sole routing algorithm
        wa_alg: identifier for a sole wavelength assignment algorithm
        rwa_alg: identifier for a routine that performs RWA as one
        ga_popsize: population size for the GA-RWA procedure
        ga_ngen: number of generations for the GA-RWA procedure
        ga_xrate: crossover rate for the GA-RWA procedure
        ga_mrate: mutation rate for the GA-RWA procedure

    Returns:
        callable: a function that combines a routing algorithm and a
            wavelength assignment algorithm if those are provided
            separately, or an all-in-one RWA procedure

    Raises:
        ValueError: if neither `rwa_alg` nor both `r_alg` and `wa_alg`
            are provided

    """

    if r_alg is not None and wa_alg is not None:
        if r_alg == 'dijkstra':
            if wa_alg == 'vertex-coloring':
                from .rwa import dijkstra_vertex_coloring
                return dijkstra_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import dijkstra_first_fit
                return dijkstra_first_fit
            elif wa_alg == 'random-fit':
                from .rwa import dijkstra_random_fit
                return dijkstra_random_fit
            else:
                raise ValueError('Unknown wavelength assignment '
                                 'algorithm "%s"' % wa_alg)
        elif r_alg == 'yen':
            if wa_alg == 'vertex-coloring':
                from .rwa import yen_vertex_coloring
                return yen_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import yen_first_fit
                return yen_first_fit
            elif wa_alg == 'random-fit':
                from .rwa import yen_random_fit
                return yen_random_fit
            else:
                raise ValueError('Unknown wavelength assignment '
                                 'algorithm "%s"' % wa_alg)
        else:
            raise ValueError('Unknown routing algorithm "%s"' % r_alg)
    elif rwa_alg is not None:
        if rwa_alg == 'genetic-algorithm':
            from .rwa import genetic_algorithm
            return genetic_algorithm(ga_popsize, ga_ngen, ga_xrate, ga_mrate)
        elif rwa_alg == "acrwa":
            from .rwa import acrwa_algorithm
            return acrwa_algorithm(net)
        elif rwa_alg == "de":
            from .rwa import de_algorithm
            return de_algorithm(net)
        elif rwa_alg == "abcrwa":
            from .rwa import abcrwa_algorithm
            return abcrwa_algorithm(net)
        else:
            raise ValueError('Unknown RWA algorithm "%s"' % rwa_alg)
    else:
        raise ValueError('RWA algorithm not specified')


def simulator(args: Namespace) -> None:
    """Main RWA simulation routine over WDM networks

    The loop levels of the simulator iterate over the number of repetitions,
    (simulations), the number of Erlangs (load), and the number of connection
    requests (calls) to be either allocated on the network or blocked if no
    resources happen to be available.

    Args:
        args: set of arguments provided via CLI to argparse module

    """
    global upper_bound
    upper_bound = 12

    # print header for pretty stdout console logging
    print('Load:   ', end='')
    for i in range(1, args.load + 1):
        print('%4d' % i, end=' ')

    time_per_simulation = []
    for simulation in range(args.num_sim):
        sim_time = default_timer()
        net = get_net_instance_from_args(args.topology, args.channels)
        rwa = get_rwa_algorithm_from_args(args.r, args.w, args.rwa,
                                          args.pop_size, args.num_gen,
                                          args.cross_rate, args.mut_rate, net)
        if args.rwa == "de":

            print("DIFFERENTIAL EVOLUTION")
            # APL, NWR, value = rwa(net, args.y) # run once since its computing a optimization value
            # print("APL", APL)
            # print("NWR", NWR)
            # print("Fitness value", value)
            # print("SHORTEST PATHS/FF")
            # APL, NWR, value = spff_algorithm(net)(net, args.y) # run once since its computing a optimization value
            # print("APL", APL)
            # print("NWR", NWR)
            # print("Fitness value", value)
            # return

            import matplotlib.pyplot as plt

            # Define lists to store data points
            net_sizes = []
            rwa_fitness_values = []
            rwa_apl_values = []
            rwa_nwr_values = []
            spff_fitness_values = []
            spff_apl_values = []
            spff_nwr_values = []

            print("start")

            # Iterate over network sizes
            for net_size in range(10, 200, 10):
                print("d", net_size, rwa_fitness_values, spff_fitness_values)
                net = DE_Graph_Custom(args.channels, net_size)

                print('starting it now')

                # RWA
                APL, NWR, rwa_value = de_algorithm(net)(net, args.y)
                rwa_fitness_values.append(rwa_value)
                rwa_apl_values.append(APL)
                rwa_nwr_values.append(NWR)

                print("done with de")

                # SPFF
                APL, NWR, spff_value = spff_algorithm(net)(net, args.y)
                spff_fitness_values.append(spff_value)
                spff_apl_values.append(APL)
                spff_nwr_values.append(NWR)

                net_sizes.append(net_size)

            # Plotting
            plt.figure(figsize=(10, 6))

            # Fitness Value Plot
            plt.subplot(2, 1, 1)
            plt.plot(net_sizes, rwa_fitness_values, label='RWA')
            plt.plot(net_sizes, spff_fitness_values, label='SPFF')
            plt.xlabel('Network Size')
            plt.ylabel('Fitness Value')
            plt.title('Fitness Value vs. Network Size for DE and SPFF Algorithms')
            plt.legend()
            plt.grid(True)

            # APL and NWR Plot
            plt.subplot(2, 1, 2)
            plt.plot(net_sizes, rwa_apl_values,
                     label='RWA APL', linestyle='--')
            plt.plot(net_sizes, rwa_nwr_values,
                     label='RWA NWR', linestyle='--')
            plt.plot(net_sizes, spff_apl_values,
                     label='SPFF APL', linestyle='-.')
            plt.plot(net_sizes, spff_nwr_values,
                     label='SPFF NWR', linestyle='-.')
            plt.xlabel('Network Size')
            plt.ylabel('Value')
            plt.title('APL and NWR vs. Network Size for DE and SPFF Algorithms')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            return
        avg_path_length_per_simulation = []

        blocklist = []
        blocks_per_erlang = []

        # ascending loop through Erlangs
        for load in range(1, args.load + 1):
            blocks = 0
            path_3_to_4_w_1 = 0
            path_3_to_4_w_2 = 0
            path_3_to_6_w_1 = 0
            path_3_to_6_w_2 = 0
            success = 0
            path_length_per_load = 0
            total_light_paths = 0
            for call in range(args.calls):
                print("call", call)
                import random

                # Uncomment for fish network
                # net.d = 7
                # net.s = random.randint(0, 2)

                # Uncomment for any other network
                net.d = random.randint(0, 12)

                while True:
                    net.s = random.randint(0, 12)
                    if net.s != net.d:
                        break

                # print('\rBlocks: ', end='', flush=True)
                # for b in blocklist:
                #     print('%04d ' % b, end='', flush=True)
                # print(' %04d' % call, end='')

                # Poisson arrival is modelled as an exponential distribution
                # of times, according to Pawełczak's MATLAB package [1]:
                # @until_next: time until the next call arrives
                # @holding_time: time an allocated call occupies net resources
                until_next = -np.log(1 - np.random.rand()) / load
                holding_time = -np.log(1 - np.random.rand())

                # Call RWA algorithm, which returns a lightpath if successful
                # or None if no λ can be found available at the route's first
                # link
                lightpath = rwa(net, args.y)

                # If lightpath is non None, the first link between the source
                # node and one of its neighbours has a wavelength available,
                # and the RWA algorithm running at that node thinks it can
                # allocate on that λ. However, we still need to check whether
                # that same wavelength is available on the remaining links
                # along the path in order to reach the destination node. In
                # other words, despite the RWA was successful at the first
                # node, the connection can still be blocked on further links
                # in the future hops to come, nevertheless.
                if lightpath is not None:
                    # check if the color chosen at the first link is available
                    # on all remaining links of the route
                    for (i, j) in lightpath.links:
                        # print("link on lightpath is", (i, j))
                        if not net.n[i][j][lightpath.w]:
                            lightpath = None
                            break

                # Check if λ was not available either at the first link from
                # the source or at any other further link along the route.
                # Otherwise, allocate resources on the network for the
                # lightpath.
                if lightpath is None:
                    blocks += 1
                else:
                    path_length_per_load += len(lightpath.r)
                    success += 1
                    total_light_paths += 1
                    if lightpath.w == 0:
                        if any([3, 4] == lightpath.r[i:i+2] for i in range(len(lightpath.r) - 1)):
                            path_3_to_4_w_1 += 1
                        elif any([3, 6] == lightpath.r[i:i+2] for i in range(len(lightpath.r) - 1)):

                            path_3_to_6_w_1 += 1

                    elif lightpath.w == 1:
                        if any([3, 4] == lightpath.r[i:i+2] for i in range(len(lightpath.r) - 1)):
                            path_3_to_4_w_2 += 1
                        elif any([3, 6] == lightpath.r[i:i+2] for i in range(len(lightpath.r) - 1)):
                            path_3_to_6_w_2 += 1

                    lightpath.holding_time = holding_time
                    net.t.add_lightpath(lightpath)
                    for (i, j) in lightpath.links:
                        net.n[i][j][lightpath.w] = 0  # lock channel
                        net.t[i][j][lightpath.w] = holding_time

                        # make it symmetric
                        net.n[j][i][lightpath.w] = net.n[i][j][lightpath.w]
                        net.t[j][i][lightpath.w] = net.t[i][j][lightpath.w]

                # FIXME The following two routines below are part of the same
                # one: decreasing the time network resources remain allocated
                # to connections, and removing finished connections from the
                # traffic matrix. This, however, should be a single routine
                # iterating over lightpaths links instead of all edges, so when
                # the time is up on all links of a lightpath, the lightpath
                # might be popped from the matrix's list. I guess the problem
                # is the random initialisation of the traffic matrix's holding
                # times during network object instantiation, but if this is
                # indeed the fact it needs some consistent testing.
                for lightpath in net.t.lightpaths:
                    if lightpath.holding_time > until_next:
                        lightpath.holding_time -= until_next
                    else:
                        # time's up: remove conn from traffic matrix's list
                        net.t.remove_lightpath_by_id(lightpath.id)

                # Update *all* channels that are still in use
                for (i, j) in net.get_edges():
                    for w in range(net.nchannels):
                        if net.t[i][j][w] > until_next:
                            net.t[i][j][w] -= until_next
                        else:
                            # time's up: free channel
                            net.t[i][j][w] = 0
                            if not net.n[i][j][w]:
                                net.n[i][j][w] = 1  # free channel

                        # make matrices symmetric
                        net.t[j][i][w] = net.t[i][j][w]
                        net.n[j][i][w] = net.n[j][i][w]
            print("TEST1", path_3_to_4_w_1)
            print("TEST2", path_3_to_4_w_2)
            print("TEST1", path_3_to_6_w_1)
            print("TEST2", path_3_to_6_w_2)
            print("successes", success)

            blocklist.append(blocks)
            blocks_per_erlang.append(100.0 * blocks / args.calls)

        sim_time = default_timer() - sim_time
        time_per_simulation.append(sim_time)
        avg_path_length_per_simulation.append(path_length_per_load/success)

        # print('\rBlocks: ', end='', flush=True)
        for b in blocklist:
            print('%04d ' % b, end='', flush=True)
        print('\n%-7s ' % 'BP (%):', end='')
        print(' '.join(['%4.1f' % b for b in blocks_per_erlang]), end=' ')
        print('[sim %d: %.2f secs]' % (simulation + 1, sim_time))

        fbase = '%s_%dch_%dreq_%s' % (
            args.rwa if args.rwa is not None else '%s_%s' % (args.r, args.w),
            args.channels, args.calls, net.name)

        write_bp_to_disk(args.result_dir, fbase + '.bp', blocks_per_erlang)
    print(avg_path_length_per_simulation)
    avg_path_length = round(sum(avg_path_length_per_simulation) /
                            len(avg_path_length_per_simulation), 2)
    print("Avg Path Length Overall All Loads", avg_path_length)

    write_it_to_disk(args.result_dir, fbase + '.it', time_per_simulation)

    if args.plot:
        plot_bp(args.result_dir)
