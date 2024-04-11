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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', required=False, default='pokec-n',
                        choices=['citeseer', 'pokec-n'],
                        help='Name of raw graph dataset')
    args = parser.parse_args()
    return args

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


def scale_attr(df, col_list):
    """
    Scale the attributes in col_list to integers starting from 0, then create an array of attributes in col_list
    If there is 1 attribute, it returns an (n, 1) array.
    :param df:
    :param col_list:
    :return: attribute array of (n, len(col_list))
    """
    n = df.shape[0]
    num_attrs = len(col_list)
    for attr_id in range(num_attrs):
        attr = col_list[attr_id]
        uniq_values = list(df[attr].unique())
        flag_not_all_int = False
        flag_has_negative = False
        flag_not_all_float = False
        for i in range(len(uniq_values)):
            is_int = isinstance(uniq_values[i], int) or isinstance(uniq_values[i], np.int64)
            is_float = isinstance(uniq_values[i], float) or isinstance(uniq_values[i], np.float64)
            flag_not_all_int = flag_not_all_int or not is_int
            flag_not_all_float = flag_not_all_float or not is_float
            if is_int or is_float:
                flag_has_negative = flag_has_negative or (uniq_values[i] < 0)

        if flag_not_all_int or flag_not_all_float or flag_has_negative:
            if flag_not_all_int:
                if not flag_not_all_float:
                    uniq_values = sorted(uniq_values)
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            else:
                uniq_values = sorted(uniq_values)
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            data = list(map(map_attr.get, df[attr]))
            df[attr] = data
    arr = df[col_list].values
    return arr


def prep_pokec(name='pokec-n', root_path='datasets'):
    ds_path = os.path.join(root_path, name)
    raw_path = os.path.join(ds_path, 'raw')
    predict_attr, sens_attr = 'I_am_working_in_field', 'region' # 'gender'
    sens_attrs = ['gender', 'region']

    idx_features_labels = pd.read_csv(os.path.join(raw_path, "region_job.csv"))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    for sens_name in sens_attrs:
        header.remove(sens_name)
    header.remove(predict_attr)

    labels = idx_features_labels[predict_attr].values # -1 means unlabeled

    # Sensitive attribute: Numpy array
    sens = scale_attr(idx_features_labels, sens_attrs)
    np.save(f'{os.path.join(ds_path, "processed")}/sens.npy', sens)
    print(f'Sensitive attributes: {sens.shape}')
    print(sens[:10])

    # Predict attribute: NumPy array
    # label_idx = np.where(labels >= 0)[0]
    # labels = labels[label_idx]
    # labels = np.append(label_idx.reshape(-1, 1), labels.reshape(-1, 1), 1).astype(np.int32)
    print(f'Labels: {labels.shape}')

    # Features: NumPy array
    features = np.array((sp.csr_matrix(idx_features_labels[header], dtype=np.float32)).todense())
    print(f'Normal attributes: {features.shape}')

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(raw_path, "region_job_relationship.txt")).astype('int')
    edge_list = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape).T
    print(f'# Edges: {edge_list.shape[1]}')

    save_processed_graph(features, labels, edge_list, ds_path)


if __name__ == '__main__':
    args = parse_args()

    data_name = args.data
    if data_name == 'citeseer':
        prep_citeseer()
    elif data_name == 'pokec-n':
        # todo: pokec-n has unlabeled nodes.
        prep_pokec(name=data_name)
    else:
        raise NotImplementedError
