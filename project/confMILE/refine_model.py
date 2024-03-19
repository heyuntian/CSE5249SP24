"""
Model for Refinement.
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from scipy.stats import entropy
from utils import graph_to_adj
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from layers import *



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_to_gcn_adj(adj, lda):  # D^{-0.5} * A * D^{-0.5} : normalized, symmetric convolution operator.
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    self_loop_wgt = np.array(adj.sum(1)).flatten() * lda  # self loop weight as much as sum. This is part is flexible.
    adj_normalized = normalize_adj(adj + sp.diags(self_loop_wgt)).tocoo()
    return adj_normalized


def convert_sparse_matrix_to_sparse_tensor(X):
    if not sp.isspmatrix_coo(X):
        X = X.tocoo()
    indices = np.mat([X.row, X.col]).transpose()
    return tf.SparseTensor(indices, X.data, X.shape)


def normalize_attr_mtx(attr_mtx, fine_graph=None):
    return softmax(attr_mtx, axis=1)
    # norm_attr_dist = attr_mtx / attr_mtx.sum(axis=1).reshape(-1, 1)
    # return norm_attr_dist


class GCN(tf.keras.Model):
    """
    Normal GCN with no fairness loss.
    """

    def __init__(self, ctrl):
        super().__init__()
        # Utils and hyperparameters
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.act_func = ctrl.refine_model.act_func
        self.wgt_decay = ctrl.refine_model.wgt_decay
        self.regularized = ctrl.refine_model.regularized
        self.learning_rate = ctrl.refine_model.learning_rate
        self.hidden_layer_num = ctrl.refine_model.hidden_layer_num
        self.lda = ctrl.refine_model.lda
        self.epoch = ctrl.refine_model.epoch
        self.early_stopping = ctrl.refine_model.early_stopping
        self.optimizer = ctrl.refine_model.tf_optimizer(learning_rate=self.learning_rate)
        self.lambda_fl = ctrl.refine_model.lambda_fl
        self.ctrl = ctrl

        # Layers
        self.conv_layers = []
        for i in range(self.hidden_layer_num):
            conv = GCNConv(self.embed_dim, activation=self.act_func, use_bias=False,
                           kernel_regularizer=regularizers.l2(l2=self.wgt_decay / 2.0) if self.regularized else None)
            self.conv_layers.append(conv)

    def call(self, gcn_A, input_embed):
        curr = input_embed
        for i in range(self.hidden_layer_num):
            curr = self.conv_layers[i]([gcn_A, curr])
        output = tf.nn.l2_normalize(curr, axis=1)
        return output

    def train(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed
        self.logger.info(f'initial_embed: {initial_embed.shape}')
        self.logger.info(f'fine_embed: {fine_embed.shape}')

        loss_arr = []
        for i in range(self.epoch):
            with tf.GradientTape() as tape:
                pred_embed = self.call(struc_A, initial_embed)
                acc_loss = tf.compat.v1.losses.mean_squared_error(fine_embed,
                                                                  pred_embed) * self.embed_dim  # tf.keras.losses.mean_squared_error(y_true=fine_embed, y_pred=pred_embed) * self.embed_dim
                loss = acc_loss
                # print(f'Epoch {i}, Loss: {loss}, Acc Loss: {acc_loss}')
                loss_arr.append(loss)
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

    def predict(self, coarse_graph=None, fine_graph=None, coarse_embed=None, last_level=False):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        initial_embed = fine_graph.C.dot(coarse_embed)
        return self.call(struc_A, initial_embed)


class GCN_conf(tf.keras.Model):
    """
    Normal GCN with no fairness loss.
    """

    def __init__(self, ctrl):
        super().__init__()
        # Utils and hyperparameters
        self.logger = ctrl.logger
        self.use_valid = ctrl.refine_model.use_valid
        self.embed_dim = ctrl.embed_dim
        self.act_func = ctrl.refine_model.act_func
        self.wgt_decay = ctrl.refine_model.wgt_decay
        self.regularized = ctrl.refine_model.regularized
        self.learning_rate = ctrl.refine_model.learning_rate
        self.hidden_layer_num = ctrl.refine_model.hidden_layer_num
        self.num_classes = ctrl.refine_model.num_classes
        self.lda = ctrl.refine_model.lda
        self.epoch = ctrl.refine_model.epoch
        self.report_epoch = ctrl.refine_model.report_epoch
        self.valid_epoch = ctrl.refine_model.valid_epoch
        self.early_stopping = ctrl.refine_model.early_stopping
        self.optimizer = ctrl.refine_model.tf_optimizer(learning_rate=self.learning_rate)
        self.lambda_fl = ctrl.refine_model.lambda_fl
        self.ctrl = ctrl

        # Layers
        self.conv_layers = []
        for i in range(self.hidden_layer_num):
            conv = GCNConv(self.embed_dim, activation=self.act_func, use_bias=False,
                           kernel_regularizer=regularizers.l2(l2=self.wgt_decay / 2.0) if self.regularized else None)
            self.conv_layers.append(conv)
        self.classifier = Dense(self.num_classes, activation='softmax')

    def call(self, gcn_A, input_embed):
        curr = input_embed
        for i in range(self.hidden_layer_num):
            curr = self.conv_layers[i]([gcn_A, curr])
        output = tf.nn.l2_normalize(curr, axis=1)
        probs = self.classifier(output)
        return output, probs

    def train(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed
        self.logger.info(f'initial_embed: {initial_embed.shape}')
        self.logger.info(f'fine_embed: {fine_embed.shape}')

        loss_arr = []
        filter_train_nodes = fine_graph.status == 0
        filter_valid_nodes = fine_graph.status == 1
        cce = SparseCategoricalCrossentropy()
        for i in range(1, self.epoch + 1):
            with tf.GradientTape() as tape:
                pred_embed, probs = self.call(struc_A, initial_embed)
                mse_loss = tf.compat.v1.losses.mean_squared_error(fine_embed,
                                                                  pred_embed) * self.embed_dim
                acc_loss = cce(fine_graph.labels[filter_train_nodes], probs[filter_train_nodes])
                loss = (1 - self.lambda_fl) * mse_loss + self.lambda_fl * acc_loss
                if i % self.report_epoch == 0:
                    print(f'Epoch {i}, Loss: {loss}, MSE Loss: {mse_loss}, ACC loss: {acc_loss}')
                if self.use_valid and i % self.valid_epoch == 0:
                    valid_acc_loss = cce(fine_graph.labels[filter_valid_nodes], probs[filter_valid_nodes])
                    print(f'\t Epoch {i}, Valid ACC loss: {valid_acc_loss}')
                loss_arr.append(loss)
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

    def predict(self, coarse_graph=None, fine_graph=None, coarse_embed=None, last_level=False):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        initial_embed = fine_graph.C.dot(coarse_embed)
        if last_level:
            return self.call(struc_A, initial_embed)
        else:
            return self.call(struc_A, initial_embed)[0]

