"""
utils
"""
import logging
import sys
from graph import Graph
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import time
'''
Packages required for link prediction datasets
'''
import os
import networkx as nx
import pickle as pkl
from typing import Dict, Tuple

def setup_custom_logger(name):
    """Set up the logger, from MILE """
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)

    return logger


class Mapping:
    """Used for mapping index of nodes since the data structure used for graph requires continuous index."""
    def __init__(self, old2new, new2old):
        self.old2new = old2new
        self.new2old = new2old


"""
read_data: Read the dataset from the disk.
"""


def read_data(args):
    graph = None
    processed_dataset_path = os.path.join('datasets', args.data, 'processed')
    assert os.path.exists(processed_dataset_path), f"Processed {args.data} dataset {processed_dataset_path} not found."
    seed = args.seed

    labels = np.load(os.path.join(processed_dataset_path, "labels.data.npy"))
    edge_list = np.load(os.path.join(processed_dataset_path, "edge_list.data.npy"))
    num_nodes = labels.shape[0]
    num_edges = edge_list.shape[1]
    labeled_nodes = np.where(labels >= 0)[0]
    num_labeled = labeled_nodes.shape[0]

    '''
    Unlabeled
    Labeled
        Label visible
            Train
            Valid
        Invisible
            Calib
            Test
    '''
    num_train, num_valid = int(num_labeled * args.train_ratio), 0
    if args.valid: # use validation set
        num_valid = int(num_train / 3)
        num_train -= num_valid
    num_invisible_label = num_labeled - num_train - num_valid
    num_calib = min(1000, num_invisible_label / 2)
    num_test = num_invisible_label - num_calib
    print(f"Data split: Total nodes {num_nodes}; Labeled {num_labeled}; Train/Valid/Calib/Test {[num_train, num_valid, num_calib, num_test]}")
    assert num_train + num_valid + num_calib + num_test == num_labeled, \
        f"Incorrect numbers of nodes. {[num_train, num_valid, num_calib, num_test, num_labeled]}; {[num_nodes]}"
    np.random.seed(seed)


    permu = np.random.permutation(num_labeled)
    status = np.ones(num_nodes) * -1 # -1 for unlabeled nodes
    status[labeled_nodes[permu[:num_train]]] = 0 # 0 for training data
    status[labeled_nodes[permu[num_train:num_train + num_valid]]] = 1 # 1 for validation data
    status[labeled_nodes[permu[num_train + num_valid:num_train + num_valid + num_calib]]] = 2 # 2 for calibration data
    status[labeled_nodes[permu[num_train + num_valid + num_calib:]]] = 3 # 3 for test data

    from collections import defaultdict
    neighbors = defaultdict(set)
    for a, b in edge_list.T:
        neighbors[a].add(b)
        if a != b:
            neighbors[b].add(a)
    num_edges = sum(len(neighbors[node_id]) for node_id in range(num_nodes))
    graph = Graph(num_nodes, num_edges)
    cnt_edges = 0
    for node_id in range(num_nodes):
        degree = graph.degree[node_id] = len(neighbors[node_id])
        graph.adj_idx[node_id + 1] = graph.adj_idx[node_id] + degree
        graph.adj_list[cnt_edges:cnt_edges + degree] = sorted(neighbors[node_id])
        cnt_edges += degree
    graph.adj_wgt = np.ones(num_edges, dtype=np.int32)
    graph.node_wgt = np.ones(num_nodes, dtype=np.int32)
    graph.labels = labels
    graph.status = status
    graph.label_range = max(labels) + 1
    return graph






def normalized(embeddings, per_feature=True):
    if per_feature:
        scaler = MinMaxScaler()
        scaler.fit(embeddings)
        return scaler.transform(embeddings)
    else:
        return normalize(embeddings, norm='l2')


def graph_to_adj(graph, self_loop=False):
    """
    self_loop: manually add self loop or not
    """
    if graph.A is not None:
        return graph.A
    node_num = graph.node_num
    adj = sp.csr_matrix((graph.adj_wgt, graph.adj_list, graph.adj_idx), shape=(node_num, node_num), dtype=np.float32)
    graph.A = adj
    # i_arr = []
    # j_arr = []
    # data_arr = []
    # for i in range(0, node_num):
    #     for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
    #         i_arr.append(i)
    #         j_arr.append(graph.adj_list[neigh_idx])
    #         data_arr.append(graph.adj_wgt[neigh_idx])
    # adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    # if self_loop:
    #     adj = adj + sp.eye(adj.shape[0])
    return adj

def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))

class Timer:
    """
    time measurement
    """
    def __init__(self, ident=0, logger=None):
        self.count = 0
        self.logger = logger
        self.ident_str = '\t' * ident
        self.restart(coldStart=True)

    def restart(self, name=None, title=None, coldStart=False):
        now = time.time()
        all_time = 0
        if not coldStart:
            self.printIntervalTime(name=title)
            all_time = now - self.startTime
            msg = "%s| Time for this section (%s): %.5f s"%(self.ident_str, name, all_time)
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
        self.startTime = now
        self.prev_time = now
        self.count = 0
        return all_time

    def printIntervalTime(self, name=None):
        now = time.time()
        msg = "%s\t| Interval %d (%s) time %.5f s"%(self.ident_str, self.count, name, now - self.prev_time)
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        self.prev_time = now
        self.count += 1



def build_test(G: nx.Graph, nodelist: Dict, ratio: float, seed=20) -> Tuple:
    """
    Split training and testing set for link prediction in graph G.
    :param G: nx.Graph
    :param nodelist: idx -> node_id in nx.Graph
    :param ratio: ratio of positive links that used for testing
    :return: Graph that remove all test edges, list of index for test edges
    """

    edges = list(G.edges.data(default=False))
    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    num_test = int(np.floor(num_edges * ratio))
    test_edges_true = []
    test_edges_false = []


    # generate false links for testing
    np.random.seed(seed)
    while len(test_edges_false) < num_test:
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
            continue
        if (idx_u, idx_v) in test_edges_false:
            continue
        else:
            test_edges_false.append((idx_u, idx_v))

    # generate true links for testing
    all_edges_idx = list(range(num_edges))
    np.random.shuffle(all_edges_idx)
    # test_edges_true_idx = all_edges_idx[:num_test]
    test_idx = 0
    while len(test_edges_true) < num_test:
        u, v, _ = edges[all_edges_idx[test_idx]]
        test_idx += 1
        if G.degree[u] <= 1 or G.degree[v] <= 1:
            continue
        G.remove_edge(u, v)
        test_edges_true.append((get_key(nodelist, u), get_key(nodelist, v)))

    # added for logistic regression
    train_edges_true = [(get_key(nodelist, u), get_key(nodelist, v)) for u, v, _ in list(G.edges.data(default=False))]
    train_edges_false = []
    while len(train_edges_false) < len(train_edges_true):
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
            continue
        if (idx_u, idx_v) in train_edges_false or (idx_u, idx_v) in test_edges_false or (idx_u, idx_v) in test_edges_true:
            continue
        else:
            train_edges_false.append((idx_u, idx_v))

    return G, test_edges_true, test_edges_false, train_edges_true, train_edges_false


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value][0]


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

