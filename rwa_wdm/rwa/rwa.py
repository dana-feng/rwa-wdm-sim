from typing import Callable, Union

from ..net import Lightpath, Network
from .routing import dijkstra, yen
from .wlassignment import vertex_coloring, first_fit, random_fit
from .ga import GeneticAlgorithm
from .acrwa import AntColonyRWA
from .de import DE, SPFF
from .abcrwa import ArtificialBeeColonyRWA
__all__ = (
    'dijkstra_vertex_coloring',
    'dijkstra_first_fit',
    'yen_vertex_coloring',
    'yen_first_fit',
    'genetic_algorithm',
)


# genetic algorithm object (global)
# FIXME this looks bad. perhaps this whole script should be a class
ga: Union[GeneticAlgorithm, None] = None


def dijkstra_vertex_coloring(net: Network, k: int) -> Union[Lightpath, None]:
    """Dijkstra and vertex coloring combination as RWA algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route = dijkstra(net.a, net.s, net.d)
    wavelength = vertex_coloring(net, Lightpath(route, None))
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def dijkstra_first_fit(net: Network, k: int) -> Union[Lightpath, None]:
    """Dijkstra and first-fit combination as RWA algorithm

    Args:
    net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route = dijkstra(net.a, net.s, net.d)
    wavelength = first_fit(net, route)
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def dijkstra_random_fit(net: Network, k: int) -> Union[Lightpath, None]:
    """Dijkstra and random-fit combination as RWA algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route = dijkstra(net.a, net.s, net.d)
    wavelength = random_fit(net, route)
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def yen_vertex_coloring(net: Network, k: int) -> Union[Lightpath, None]:
    """Yen and vertex coloring combination as RWA algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    routes = yen(net.a, net.s, net.d, k)
    for route in routes:
        wavelength = vertex_coloring(net, Lightpath(route, None))
        if wavelength is not None and wavelength < net.nchannels:
            return Lightpath(route, wavelength)
    return None


def yen_first_fit(net: Network, k: int) -> Union[Lightpath, None]:
    """Yen and first-fit combination as RWA algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    routes = yen(net.a, net.s, net.d, k)
    for route in routes:
        wavelength = first_fit(net, route)
        if wavelength is not None and wavelength < net.nchannels:
            return Lightpath(route, wavelength)
    return None


def yen_random_fit(net: Network, k: int) -> Union[Lightpath, None]:
    """Yen and random-fit combination as RWA algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    routes = yen(net.a, net.s, net.d, k)
    for route in routes:
        wavelength = random_fit(net, route)
        if wavelength is not None and wavelength < net.nchannels:
            return Lightpath(route, wavelength)
    return None


def genetic_algorithm_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via genetic algorithm

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route, wavelength = ga.run(net, k)
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def genetic_algorithm(pop_size: int, num_gen: int,
                      cross_rate: float, mut_rate: float) -> Callable:
    """Genetic algorithm as both routing and wavelength assignment algorithm

    This function just sets the parameters to the GA, so it acts as if it were
    a class constructor, setting a global variable as instance to the
    `GeneticAlgorithm` object in order to be further used by a callback
    function, which in turn returns the lightpath itself upon RWA success. This
    split into two classes is due to the fact that the class instance needs to
    be executed only once, while the callback may be called multiple times
    during simulation, namely one time per number of arriving call times number
    of load in Erlags (calls * loads)

    Note:
        Maybe this entire script should be a class and `ga` instance could be
        an attribute. Not sure I'm a good programmer.

    Args:
        pop_size: number of chromosomes in the population
        num_gen: number of generations towards evolve
        cross_rate: percentage of individuals to perform crossover
        mut_rate: percentage of individuals to undergo mutation

    Returns:
        callable: a callback function that calls the `GeneticAlgorithm` runner
            class, which finally and properly performs the RWA procedure

    """
    global ga
    ga = GeneticAlgorithm(pop_size, num_gen, cross_rate, mut_rate)
    return genetic_algorithm_callback


def acrwa_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via acrwa

    Args:
        net: Network topology instance

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route, wavelength = acrwa_algo.run(net, k)
    if wavelength is not None and wavelength < net.nchannels and route:
        return Lightpath(route, wavelength)
    return None


def abcrwa_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via ABCRWA

    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    route, wavelength = abcrwa_algo.run(net)
    if wavelength is not None and wavelength < net.nchannels and route:
        return Lightpath(route, wavelength)
    return None


def acrwa_algorithm(net: Network) -> Callable:
    global acrwa_algo
    acrwa_algo = AntColonyRWA()
    acrwa_algo.initialization(net)
    return acrwa_callback


def de_algorithm(net: Network) -> Callable:
    global de_algo
    de_algo = DE(net)
    return de_callback


def abcrwa_algorithm(net: Network) -> Callable:
    global abcrwa_algo
    abcrwa_algo = ArtificialBeeColonyRWA()
    abcrwa_algo.initialization(net)
    return abcrwa_callback


def de_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via acrwa

    Args:
        net: Network topology instance

    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath

    """
    result = de_algo.run(net, k)
    return result


def spff_algorithm(net: Network) -> Callable:
    global spff_algo
    spff_algo = SPFF(net)
    return spff_callback


def spff_callback(net, k):
    result = spff_algo.run(net, k)
    return result
