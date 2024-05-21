import heapq
import importlib
import math
import time
import torch
import pandas as pd

torch.multiprocessing.set_start_method('spawn')
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import multiprocessing as mp
from functools import lru_cache

import networkit as nk
import networkx as nx
import numpy as np
import ot

from .util import logger, set_verbose, cut_graph_by_cutoff, get_rf_metric_cutoff

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.5
_weight = "weight"
_method = "OTDSinkhornMix"
_base = math.e
_exp_power = 2
_proc = mp.cpu_count()
_cache_maxsize = 1000000
_shortest_path = "all_pairs"
_nbr_topk = 3000
_OTDSinkhorn_threshold = 2000
_apsp = {}


# -------------------------------------------------------

@lru_cache(_cache_maxsize)
def _get_single_node_neighbors_distributions(node, direction="successors"):
    """
    Get the neighbor density distribution of given node `node`.
    """
    if _Gk.isDirected():
        if direction == "predecessors":
            neighbors = list(_Gk.iterInNeighbors(node))
        else:  # successors
            neighbors = list(_Gk.iterNeighbors(node))
    else:
        neighbors = list(_Gk.iterNeighbors(node))

    # Get sum of distributions from x's all neighbors
    heap_weight_node_pair = []
    for nbr in neighbors:
        if direction == "predecessors":
            w = _base ** (-_Gk.weight(nbr, node) ** _exp_power)
        else:  # successors
            w = _base ** (-_Gk.weight(node, nbr) ** _exp_power)

        if len(heap_weight_node_pair) < _nbr_topk:
            heapq.heappush(heap_weight_node_pair, (w, nbr))
        else:
            heapq.heappushpop(heap_weight_node_pair, (w, nbr))

    nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

    if not neighbors:
        # No neighbor, all mass stay at node
        return [1], [node]

    if nbr_edge_weight_sum > EPSILON:
        # Sum need to be not too small to prevent divided by zero
        distributions = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]
    else:
        # Sum too small, just evenly distribute to every neighbors
        logger.warning("Neighbor weight sum too small, list:", heap_weight_node_pair)
        distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)

    nbr = [x[1] for x in heap_weight_node_pair]
    return distributions + [_alpha], nbr + [node]


def _distribute_densities(source, target):
    """
    Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors. Notice that only neighbors with top `_nbr_topk` edge weights.
    """

    # Distribute densities for source and source's neighbors as x
    t0 = time.time()

    if _Gk.isDirected():
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "predecessors")
    else:
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "successors")

    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target, "successors")

    logger.debug("%8f secs density distribution for edge." % (time.time() - t0))

    # construct the cost dictionary from x to y
    t0 = time.time()

    if _shortest_path == "pairwise":
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(_source_target_shortest_path(src, tgt))
            d.append(tmp)
        d = np.array(d)
    else:  # all_pairs
        d = _apsp[np.ix_(source_topknbr, target_topknbr)]  # transportation matrix

    x = np.array(x)  # the mass that source neighborhood initially owned
    y = np.array(y)  # the mass that target neighborhood needs to received

    logger.debug("%8f secs density matrix construction for edge." % (time.time() - t0))

    return x, y, source_topknbr, target_topknbr, d


@lru_cache(_cache_maxsize)
def _source_target_shortest_path(source, target):
    """
    Compute pairwise shortest path from `source` to `target` by BidirectionalDijkstra via Networkit.
    """

    length = nk.distance.BidirectionalDijkstra(_Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length


def _get_all_pairs_shortest_path():
    """
    Pre-compute all pairs shortest paths of the assigned graph `_Gk`.
    """
    logger.trace("Start to compute all pair shortest path.")

    global _Gk

    t0 = time.time()
    apsp = nk.distance.APSP(_Gk).run().getDistances()
    logger.trace("%8f secs for all pair by NetworKit." % (time.time() - t0))

    return np.array(apsp)


def _parse_to_tensor(x, y, d):
    x = torch.Tensor(x)  # .to(_device)
    y = torch.Tensor(y)  # .to(_device)
    d = torch.Tensor(d)

    return x, y, d


def _optimal_transportation_distance(x, y, d):
    """
    Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.
    """
    t0 = time.time()
    m = ot.emd2(x, y, d)
    logger.debug(
        "%8f secs for Wasserstein dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _sinkhorn_distance(x, y, d):
    """
    Compute the approximate optimal transportation distance (Sinkhorn distance) of the given density distributions.
    """
    x, y = _parse_to_tensor(x, y)
    t0 = time.time()
    m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')
    logger.debug(
        "%8f secs for Sinkhorn dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _average_transportation_distance(source, target):
    """
    Compute the average transportation distance (ATD) of the given density distributions.
    """

    t0 = time.time()
    if _Gk.isDirected():
        source_nbr = list(_Gk.iterInNeighbors(source))
    else:
        source_nbr = list(_Gk.iterNeighbors(source))
    target_nbr = list(_Gk.iterNeighbors(target))

    share = (1.0 - _alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = _alpha * _apsp[source][target]

    for src in source_nbr:
        for tgt in target_nbr:
            cost_nbr += _apsp[src][tgt] * share

    m = cost_nbr + cost_self  # Average transportation cost

    logger.debug("%8f secs for avg trans. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0,
                                                                                       len(source_nbr),
                                                                                       len(target_nbr)))
    return m


def _compute_ricci_curvature_single_edge(source, target):
    """
    Ricci curvature computation for a given single edge.
    """
    # logger.debug("EDGE:%s,%s"%(source,target))
    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if _Gk.weight(source, target) < EPSILON:
        logger.trace("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                     (source, target))
        return {(source, target): 0}

    # compute transportation distance
    m = 1  # assign an initial cost
    assert _method in ["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix"], \
        'Method %s not found, support method:["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix]' % _method

    x, y, neighbors_x, neighbors_y, d = _distribute_densities(source, target)
    optimal_plan = ot.emd(x, y, d)
    optimal_cost = optimal_plan * d
    optimal_total_cost = np.sum(optimal_cost)
    optimal_cost = pd.DataFrame(optimal_cost, columns=neighbors_y, index=neighbors_x)

    '''
    if _method == "OTD":
        x, y, d = _distribute_densities(source, target)
        m = _optimal_transportation_distance(x, y, d)
    elif _method == "ATD":
        m = _average_transportation_distance(source, target)
    elif _method == "Sinkhorn":
        x, y, d = _distribute_densities(source, target)
        m = _sinkhorn_distance(x, y, d)
    elif _method == "OTDSinkhornMix":
        x, y, d = _distribute_densities(source, target)
        # When x and y are small (usually around 2000 to 3000), ot.emd2 is way faster than ot.sinkhorn2
        # So we only do sinkhorn when both x and y are too large for ot.emd2
        if len(x) > _OTDSinkhorn_threshold and len(y) > _OTDSinkhorn_threshold:
            m = _sinkhorn_distance(x, y, d)
        else:
            m = _optimal_transportation_distance(x, y, d)
    '''

    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (optimal_total_cost / _Gk.weight(source, target))  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {
        (source, target): {
            'rc_curvature': result,
            'rc_transport_cost': optimal_cost,
        }
    }


def _wrap_compute_single_edge(stuff):
    """
    Wrapper for args in multiprocessing.
    """
    return _compute_ricci_curvature_single_edge(*stuff)


def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000):
    """
    Compute Ricci curvature for edges in  given edge lists.
    """

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        # if not _apsp:
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    with mp.get_context('fork').Pool(processes=_proc) as pool:
        # WARNING: Now only fork works, spawn will hang.

        # Decide chunksize following method in map_async
        if chunksize is None:
            chunksize, extra = divmod(len(args), proc * 4)
            if extra:
                chunksize += 1
            if chunksize == 0: chunksize = 1

        # Compute Ricci curvature for edges
        result = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()

    # Convert edge index from nk back to nx for final output
    output = {}
    for rc in result:
        for k in list(rc.keys()):
            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output


def _compute_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):
    """
    Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.
    """

    # compute Ricci curvature for all edges
    edge_ricci = _compute_ricci_curvature_edges(G, weight=weight, **kwargs)

    # Assign edge Ricci curvature from result to graph G
    nx.set_edge_attributes(G, edge_ricci, "ricciCurvature")

    # Compute node Ricci curvature
    for n in G.nodes():
        rc_sum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rc_sum += G[n][nbr]['ricciCurvature']['rc_curvature']

            # Assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rc_sum / G.degree(n)
            logger.debug("node %s, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    return G


def _compute_ricci_flow(G: nx.Graph, weight="weight",
                        iterations=20, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100),
                        **kwargs
                        ):
    """
    Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.
    """

    if not nx.is_connected(G):
        logger.info("Not connected graph detected, compute on the largest connected component instead.")
        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))

    # Set normalized weight to be the number of edges.
    normalized_weight = float(G.number_of_edges())

    global _apsp

    # Start compute edge Ricci flow
    t0 = time.time()

    if nx.get_edge_attributes(G, "original_RC"):
        logger.info("original_RC detected, continue to refine the ricci flow.")
    else:
        logger.info("No ricciCurvature detected, compute original_RC...")
        _compute_ricci_curvature(G, weight=weight, **kwargs)

        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

        # clear the APSP since the graph have changed.
        _apsp = {}

    # Start the Ricci flow process
    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2][weight] -= step * (G[v1][v2]["ricciCurvature"]) * G[v1][v2][weight]

        # Do normalization on all weight to prevent weight expand to infinity
        w = nx.get_edge_attributes(G, weight)
        sumw = sum(w.values())
        for k, v in w.items():
            w[k] = w[k] * (normalized_weight / sumw)
        nx.set_edge_attributes(G, values=w, name=weight)
        logger.info(" === Ricci flow iteration %d === " % i)

        _compute_ricci_curvature(G, weight=weight, **kwargs)

        rc = nx.get_edge_attributes(G, "ricciCurvature")
        diff = max(rc.values()) - min(rc.values())

        logger.trace("Ricci curvature difference: %f" % diff)
        logger.trace("max:%f, min:%f | maxw:%f, minw:%f" % (
            max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

        if diff < delta:
            logger.trace("Ricci curvature converged, process terminated.")
            break

        # do surgery or any specific evaluation
        surgery_func, do_surgery = surgery
        if i != 0 and i % do_surgery == 0:
            G = surgery_func(G, weight)
            normalized_weight = float(G.number_of_edges())

        for n1, n2 in G.edges():
            logger.debug("%s %s %s" % (n1, n2, G[n1][n2]))

        # clear the APSP since the graph have changed.
        _apsp = {}

    logger.info("%8f secs for Ricci flow computation." % (time.time() - t0))

    return G


class OllivierRicci:
    """
    A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
    Node Ricci curvature is defined as the average of all it's adjacency edge.
    """

    def __init__(self, G: nx.Graph, weight="weight", alpha=0.5, method="OTDSinkhornMix",
                 base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, shortest_path="all_pairs",
                 cache_maxsize=1000000,
                 nbr_topk=3000, verbose="ERROR"):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc
        self.chunksize = chunksize
        self.cache_maxsize = cache_maxsize
        self.shortest_path = shortest_path
        self.nbr_topk = nbr_topk

        self.set_verbose(verbose)
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert importlib.util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

        if not nx.get_edge_attributes(self.G, weight):
            logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.info('Self-loop edge detected. Removing %d self-loop edges.' % len(self_loop_edges))
            self.G.remove_edges_from(self_loop_edges)

    def set_verbose(self, verbose):
        """
        Set the verbose level for this process.
        """
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """
        Compute Ricci curvature for edges in given edge lists.
        """
        return _compute_ricci_curvature_edges(G=self.G, weight=self.weight, edge_list=edge_list,
                                              alpha=self.alpha, method=self.method,
                                              base=self.base, exp_power=self.exp_power,
                                              proc=self.proc, chunksize=self.chunksize,
                                              cache_maxsize=self.cache_maxsize, shortest_path=self.shortest_path,
                                              nbr_topk=self.nbr_topk)

    def compute_ricci_curvature(self):
        """
        Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.
        """

        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight,
                                          alpha=self.alpha, method=self.method,
                                          base=self.base, exp_power=self.exp_power,
                                          proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                          shortest_path=self.shortest_path,
                                          nbr_topk=self.nbr_topk)
        return self.G

    def compute_ricci_flow(self, iterations=10, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """
        Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.
        """
        self.G = _compute_ricci_flow(G=self.G, weight=self.weight,
                                     iterations=iterations, step=step, delta=delta, surgery=surgery,
                                     alpha=self.alpha, method=self.method,
                                     base=self.base, exp_power=self.exp_power,
                                     proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                     shortest_path=self.shortest_path, nbr_topk=self.nbr_topk)
        return self.G

    def ricci_community(self, cutoff_step=0.025, drop_threshold=0.01):
        """
        Detect community clustering by Ricci flow metric.
        The communities are detected by the modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.
        """

        cc = self.ricci_community_all_possible_clusterings(cutoff_step=cutoff_step, drop_threshold=drop_threshold)
        assert cc, "No clustering found!"

        number_of_clustering = len(set(cc[-1][1].values()))
        logger.info("Communities detected: %d" % number_of_clustering)

        return cc[-1]

    def ricci_community_all_possible_clusterings(self, cutoff_step=0.025, drop_threshold=0.01):
        """
        Detect community clustering by Ricci flow metric (all possible clustering guesses).
        The communities are detected by Modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.
        """

        if not nx.get_edge_attributes(self.G, "original_RC"):
            logger.info("Ricci flow not detected yet, run Ricci flow with default setting first...")
            self.compute_ricci_flow()

        logger.info("Ricci flow detected, start cutting graph into community...")
        cut_guesses = \
            get_rf_metric_cutoff(self.G, weight=self.weight, cutoff_step=cutoff_step, drop_threshold=drop_threshold)
        assert cut_guesses, "No cutoff point found!"

        Gp = self.G.copy()
        cc = []
        for cut in cut_guesses[::-1]:
            Gp = cut_graph_by_cutoff(Gp, cutoff=cut, weight=self.weight)
            # Get connected component after cut as clustering
            cc.append((cut, {c: idx for idx, comp in enumerate(nx.connected_components(Gp)) for c in comp}))

        return cc
