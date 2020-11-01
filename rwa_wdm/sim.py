"""RWA WDM: Routing and Wavelength Assignment Simulator over WDM Networks

This is a simulator for RWA algorithms over wavelength-multiplexed, all-optical
networks with static traffic. It implements some mainstream algorithms for both
routing and wavelength assignment subproblems, such as Dijkstra and Yen, and
first-fit and vertex-coloring, respectively.

Besides, a self-made genetic algorithms is also implemented to solve the RWA
problem.

"""

# [1] https://la.mathworks.com/matlabcentral/fileexchange/4797-wdm-network-blocking-computation-toolbox

import sys
import os
import logging
from typing import Callable

import numpy as np

from .io import write_results_to_disk, plot_blocking_probability
from .net import Network

logger = logging.getLogger(__name__)


def get_net_instance_from_args(topname: str, numch: int) -> Network:
    """Parse args and return a Network topology instance

    """
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
    else:
        raise ValueError('No network named "%s"' % topname)


def get_rwa_algorithm_from_args(r_alg: str, wa_alg: str,
                                rwa_alg: str) -> Callable:
    """Parse args and returns a function that performs rwa

    """
    if r_alg is not None:  # NOTE implies `wa_alg` is not `None`
        if r_alg == 'dijkstra':
            if wa_alg == 'vertex-coloring':
                from .rwa import dijkstra_vertex_coloring
                return dijkstra_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import dijkstra_first_fit
                return dijkstra_first_fit
            else:
                raise ValueError('Unknown algorithm "%s"' % wa_alg)
        elif r_alg == 'yen':
            if wa_alg == 'vertex-coloring':
                from .rwa import yen_vertex_coloring
                return yen_vertex_coloring
            elif wa_alg == 'first-fit':
                from .rwa import yen_first_fit
                return yen_first_fit
            else:
                raise ValueError('Unknown algorithm "%s"' % wa_alg)
        else:
            raise ValueError('Unknown algorithm "%s"' % r_alg)
    elif rwa_alg is not None:
        if rwa_alg == 'ga':
            from .rwa import genetic_algorithm
            return genetic_algorithm
        else:
            raise ValueError('Unknown algorithm "%s"' % rwa_alg)
    else:
        raise ValueError('Algorithm not specified')


def rwa_simulator(args):

    # TODO parse args by group
    net = get_net_instance_from_args(args.topology, args.channels)
    rwa = get_rwa_algorithm_from_args(args.r, args.w, args.rwa_alg)

    blocks_per_erlang = []

    # print header for pretty stdout console logging
    print('Load:   ', end='')
    for i in range(1, args.load + 1):
        print('%4d' % i, end=' ')
    print('\nBlocks: ', end='')

    # ascending loop through Erlangs
    for load in range(1, args.load + 1):
        blocks = 0
        for call in range(args.calls):
            # Poisson arrival is here modelled as an exponential distribution
            # of times, according to Pawełczak's MATLAB package [1]:
            # @until_next: time until the next call arrives
            # @holding_time: time an allocated call occupies net resources
            until_next = -np.log(1 - np.random.rand()) / load
            holding_time = -np.log(1 - np.random.rand())

            # Call RWA algorithm, which returns a lightpath if successful or
            # None if no λ can be found available at the route's first link
            lightpath = rwa(net, args.y)

            # If lightpath is non None, the first link between the source node
            # and one of its neighbours has a wavelength available, and the RWA
            # algorithm running at that node thinks it can allocate on that λ.
            # However, we still need to check whether that same wavelength is
            # available on the remaining links along the path in order to reach
            # the destination node. In other words, despite the RWA was
            # successful at the first node, the connection can still be blocked
            # on further links in the future hops to come, nevertheless.
            if lightpath is not None:
                # check if the color chosen at the first link is available on
                # all remaining links of the route
                for (i, j) in lightpath.links:
                    if not net.get_wave_availability(lightpath.w, net.n[i][j]):
                        lightpath = None
                        break

            # Check if λ was not available either at the first link from the
            # source or at any other further link along the route. Otherwise,
            # we allocate resources on the network for the lightpath.
            if lightpath is None:
                blocks += 1
            else:
                lightpath.holding_time = holding_time
                net.t.add_lightpath(lightpath)
                for (i, j) in lightpath.links:
                    net.n[i][j] -= 2 ** lightpath.w  # lock channel (bit: 0)
                    net.t[i][j][lightpath.w] = holding_time

                    # make it symmetric
                    net.n[j][i] = net.n[i][j]
                    net.t[j][i][lightpath.w] = net.t[i][j][lightpath.w]

            # FIXME The following two routines below are part of the same one:
            # decreasing the time network resources remain allocated to
            # connections, and removing finished connections from the traffic
            # matrix. This, however, should be a single routine iterating over
            # lightpaths links instead of all edges, so when the time is up on
            # all links of a lightpath, the lightpath might be popped from the
            # matrix's list. I guess the problem is the random initialisation
            # of the traffic matrix's holding times during network object
            # instantiation, but if this is indeed the fact it needs some
            # consistent testing.
            for lightpath in net.t.lightpaths:
                if lightpath.holding_time > until_next:
                    lightpath.holding_time -= until_next
                else:
                    # time's up: remove connection from traffic matrix's list
                    net.t.remove_lightpath_by_id(lightpath.id)

            # Update *all* channels that are still in use
            for (i, j) in net.get_edges():
                for w in range(net._num_channels):  # FIXME
                    if net.t[i][j][w] > until_next:
                        net.t[i][j][w] -= until_next
                    else:
                        # time's up: free channel
                        net.t[i][j][w] = 0
                        if not net.get_wave_availability(w, net.n[i][j]):
                            net.n[i][j] += 2 ** w  # free channel (bit: 1)

                    # make matrices symmetric
                    net.t[j][i][w] = net.t[i][j][w]
                    net.n[j][i] = net.n[j][i]

        print('%04d ' % blocks, end='', flush=True)
        blocks_per_erlang.append(100.0 * blocks / args.calls)

    print('\n%-7s ' % net.name, end='')
    print(' '.join(['%4.1f' % b for b in blocks_per_erlang]))

    write_results_to_disk(args.result_dir, net.name + '.bp', blocks_per_erlang)
    if args.plot:
        plot_blocking_probability(args.result_dir)