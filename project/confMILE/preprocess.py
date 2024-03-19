#!/usr/bin/env python
"""
Preprocess the data into a uniform format
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import _pickle as cPickle
import networkx as nx


def save_processed_graph(features, labels, edge_list, dataset_path):
    processed_path = os.path.join(dataset_path, 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    np.save(os.path.join(processed_path, "features.data"), features)
    np.save(os.path.join(processed_path, "labels.data"), labels)
    np.save(os.path.join(processed_path, "edge_list.data"), edge_list)


def prep_citeseer(ds_path='datasets/citeseer'):
    r_path = os.path.join(ds_path, "raw/")

    # load the original splits
    with open(os.path.join(r_path, "ind.citeseer.graph"), "rb") as f:
        graph_adj = cPickle.load(f)

    with open(os.path.join(r_path, "ind.citeseer.test.index")) as f:
        test_inds = [int(x) for x in f.readlines()]

    with open(os.path.join(r_path, "ind.citeseer.allx"), "rb") as f:
        trainval_feats = cPickle.load(f, encoding="latin1")

    with open(os.path.join(r_path, "ind.citeseer.ally"), "rb") as f:
        trainval_labels = cPickle.load(f, encoding="latin1")

    with open(os.path.join(r_path, "ind.citeseer.tx"), "rb") as f:
        test_feats = cPickle.load(f, encoding="latin1")

    with open(os.path.join(r_path, "ind.citeseer.ty"), "rb") as f:
        test_labels = cPickle.load(f, encoding="latin1")

    # create combined graph
    # stack trainval and test feats intersepersed using train_inds and test_inds
    X = sp.vstack((trainval_feats, test_feats))
    y = np.vstack((trainval_labels, test_labels))
    # assert single label
    assert max(y.sum(axis=1)) == 1

    G = nx.from_dict_of_lists(graph_adj)
    # TODO fix the node indices (following datasets/utils.py)
    # is this correct?
    test_idx_range = np.sort(test_inds)
    missing_idx = set(range(min(test_idx_range), max(test_idx_range) + 1)) - set(test_idx_range)
    for idx in missing_idx:
        G.remove_node(idx)

    nodes = sorted(G.nodes())
    node2idx = {node: idx for idx, node in zip(range(G.number_of_nodes()), list(nodes))}
    # TODO: sparse storage does not work
    # features = csr_matrix_to_torch_sparse_tensor(X)
    features = X.todense()
    labels = y.argmax(axis=1)
    edge_list = np.array([(node2idx[u], node2idx[v]) for u, v in G.edges()]).T

    save_processed_graph(features, labels, edge_list, ds_path)

if __name__ == '__main__':
    prep_citeseer()