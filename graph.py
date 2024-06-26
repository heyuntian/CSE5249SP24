"""
graph.py: Data structure of the graph

@author: Yuntian He
"""

import numpy as np


class Graph(object):

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # number of nodes
        self.edge_num = edge_num  # number of edges
        self.adj_list = np.zeros(edge_num, dtype=np.int32)  # adjacency list
        self.adj_wgt = np.zeros(edge_num, dtype=np.float32)  # weight of each edge
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # index of the beginning neighbor in adj_list of each# node
        self.node_wgt = np.zeros(node_num, dtype=np.float32)    # weight of each node
        self.degree = np.zeros(node_num, dtype=np.float32)   # sum of incident edges' weights of each node
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1
        self.labels = np.zeros(node_num, dtype=np.int32)
        self.status = np.zeros(node_num, dtype=np.int32)
        self.label_range = 0
        self.sens = None


        self.coarser = None
        self.finer = None
        self.C = None
        self.A = None

    def resize_adj(self, edge_num):
        """ Resize the adjacency list/wgts based on the number of edges."""
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        """obtain the list of neigbors given a node."""
        idx_start = self.adj_idx[idx]
        idx_end = self.adj_idx[idx + 1]
        return self.adj_list[idx_start:idx_end]

    def get_neigh_edge_wgts(self, idx):
        """obtain the weights of neighbors given a node."""
        idx_start = self.adj_idx[idx]
        idx_end = self.adj_idx[idx + 1]
        return self.adj_wgt[idx_start:idx_end]
