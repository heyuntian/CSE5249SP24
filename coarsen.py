"""
Matching methods for graph coarsening.
"""
import numpy as np
from graph import Graph
from utils import cmap2C
# from similarity import *
from collections import defaultdict
from scipy.stats import entropy


def normalized_adj_wgt(ctrl, graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree  # sum of incident edges' weights of each node
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt


def mile_match(ctrl, graph):
    """
    Matching method in MILE, with no fairness considered
    :param ctrl:
    :param graph:
    :return:
    """
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(ctrl, graph)

    max_node_wgt = ctrl.max_node_wgt

    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(ctrl, graph, matched, groups)
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if (not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt:
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node.
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return groups, coarse_graph_size


def is_Consistent_Status(i: int, j: int):
    if i >= 2: # calib or test
        return j >= 2
    return i == j

def confmile_match(ctrl, graph):
    """
    Matching method in Conformal MILE
    :param ctrl:
    :param graph:
    :return:
    """
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(ctrl, graph)
    max_node_wgt = ctrl.max_node_wgt

    groups = []
    matched = [False] * node_num

    # SEM
    jaccard_idx_preprocess(ctrl, graph, matched, groups, is_conformal=True)
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))

    # NHEM
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(node_num)]
    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        node_status = graph.status[idx]
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:
                continue
            curr_wgt = norm_adj_wgt[j]
            if not matched[neigh] and is_Consistent_Status(node_status, graph.status[neigh]) \
                and max_wgt < curr_wgt \
                and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt:
                max_idx, max_wgt = neigh, curr_wgt
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return groups, coarse_graph_size

def jaccard_idx_preprocess(ctrl, graph, matched, groups, is_conformal=False):
    """
    Structure-Equivalent Matching in MILE (Liang et al, 2021)
    Use hashmap to find out nodes with exactly same neighbors.
    :param ctrl:
    :param graph:
    :param matched:
    :param groups:
    :return:
    """
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        if is_conformal:
            neighs = neighs + f'-{graph.status[i]}'
        neighs2node[neighs].append(i)
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def create_coarse_graph(ctrl, graph, groups, coarse_graph_size):
    """
    Create the coarsened graph of Conformal MILE.
    :param ctrl:
    :param graph:
    :param groups:
    :param coarse_graph_size:
    :return:
    """
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
    labels = graph.labels
    status = graph.status

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree
    coarse_labels = coarse_graph.labels
    coarse_status = coarse_graph.status

    coarse_adj_idx[0] = 0
    nedges = 0
    for idx in range(len(groups)):
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        coarse_labels[coarse_node_idx] = labels[group[0]]
        coarse_status[coarse_node_idx] = status[group[0]]

        for merged_node in group:
            coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]
            for j in range(adj_idx[merged_node], adj_idx[merged_node + 1]):
                k = cmap[adj_list[j]]
                if k not in neigh_dict:
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                coarse_degree[coarse_node_idx] += adj_wgt[j]
        coarse_adj_idx[coarse_node_idx + 1] = nedges
    coarse_graph.edge_num = nedges
    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)
    graph.C = C
    return coarse_graph

"""
Define your attribute divergence functions here.
You need to manually add them to argument parser in `main.py`. 
"""


def kl_dvg(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    """
    Compute the KL-divergence of two normalized attribute vector.
    Note that the attribute vectors must be normalized (sum to 1) before calling it.
    """
    return 1 - 1 / (1 + entropy(norm1, norm2))


def mergefair_group(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    return attr2.dot(wgt1 - attr1) / sens_num / (wgt1 * wgt2)


def abs_diff(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    return np.sum(np.abs(attr1 - attr2)) / sens_num / (wgt1 + wgt2)
