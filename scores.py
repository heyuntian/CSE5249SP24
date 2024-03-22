from abc import ABC
import math

import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import TQDMProgressBar
#
# from models import GNN, CFGNN
# from utils import dl_affinity_setup

class CPScore(ABC):
    def __init__(self, **kwargs):
        self.defined_args = kwargs
    
    def pipe_compute(self, probs):
        return self.compute(probs, **self.defined_args)

    def compute(self, probs, **kwargs):
        return probs
    
    def compute_quantile(self, scores, alpha):
        n = scores.shape[0]
        return torch.quantile(
            scores, min(1, math.ceil((n + 1) * (1 - alpha)) / n), interpolation='higher'
        )

class APSScore(CPScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_eps = kwargs.get("use_aps_epsilon", True)

    def compute(self, probs, **kwargs):
        # a vectorized implementation of APS score from
        # . https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/cp/transformations.py
        # sorted probs: n_samples x n_classes

        probs_pi_rev_indices = torch.argsort(-probs, dim=1)
        sorted_probs_pi = torch.take_along_dim(probs, probs_pi_rev_indices, dim=1)
        # PI[i, j] = sum(pi_(1) + pi_(2) + ... + pi_(j-1))
        # PI[i, 0] = 0
        PI = torch.zeros((sorted_probs_pi.shape[0], sorted_probs_pi.shape[1] + 1),
                         device=probs.device)
        PI[:, 1:] = torch.cumsum(sorted_probs_pi, dim=1)
        # we vectorize this loop
        #ranks = torch.zeros((n_samples, n_classes), dtype=torch.int32)
        #for i in range(n_samples):
        #    ranks[i, sorted_order[i]] = torch.arange(n_classes -1, -1, -1)
        ranks = probs_pi_rev_indices.argsort(dim=1)
        
        # cumulative score up to rank j
        # cls_scores[i, j] = NC score for class j for sample i
        # that is assuming that the true class is j
        # cls_score[i, j] = PI[i, rank[j]] + (1 - u) * probs[i, j]
        # note that PI starts at 0, so PI[i, rank[j]] = sum(probs[:rank[j] - 1])
        if self.use_eps:
            # whether to use uniform noise to adjust set size
            u_vec = torch.rand_like(probs) # u_vec[i, j] = u for sample i cls j  
            cls_scores = PI.gather(1, ranks) + (1 - u_vec) * probs
        else:
            cls_scores = PI.gather(1, ranks + 1)
        cls_scores = torch.min(cls_scores, torch.ones_like(cls_scores))
        return cls_scores
    
    def construct_conformal_set(self, probs, taus, **kwargs):
        pass


# class CFGNNScore(CPScore):
#     def __init__(self, base_model_path, use_aps_epsilon: bool = True,
#                  **kwargs):
#         super().__init__(use_aps_epsilon=use_aps_epsilon, **kwargs)
#
#         self.trainable_model = CFGNN(
#             backbone=kwargs.get("backbone"),
#             in_feats=kwargs.get("num_classes"),
#             h_feats=kwargs.get("confgnn_hidden_channels"),
#             num_classes=kwargs.get("num_classes"),
#             base_model_path=base_model_path,
#             alpha=kwargs.get("alpha", 0.1),
#             score_fn=APSScore(use_aps_epsilon=use_aps_epsilon, **kwargs),
#             num_confgnn_heads=kwargs.get("confgnn_heads", 1),
#             confgnn_aggr=kwargs.get("confgnn_aggr", "mean"),
#             num_confgnn_layers=kwargs.get("confgnn_layers", 2),
#             confgnn_lr=kwargs.get("confgnn_lr", 0.01)
#         )
#
#         #self.ckpt_dir = kwargs.get("ckpt_dir")
#         devices = self.defined_args.get("num_gpus", 1)
#         num_nodes = self.defined_args.get("num_nodes", 1)
#         max_epochs = kwargs.get("confgnn_epochs", 1)
#         self.pt = pl.Trainer(
#             accelerator="gpu",
#             devices=devices,
#             num_nodes=num_nodes,
#             max_epochs=max_epochs,
#             callbacks = [
#                 TQDMProgressBar(refresh_rate=100),
#             ],
#             log_every_n_steps=10,
#             check_val_every_n_epoch=1
#         )
#
#     def compute(self, dl, **kwargs):
#         with dl_affinity_setup(dl)():
#             with torch.no_grad():
#                 self.pt.test(
#                     self.trainable_model,
#                     dataloaders=dl
#                 )
#                 scores, labels = self.trainable_model.latest_test_results
#
#         return scores, labels
#
#     def learn_params(self, calib_dl):
#         with dl_affinity_setup(calib_dl)():
#             # first fit the model
#             self.pt.fit(
#                 self.trainable_model,
#                 train_dataloaders=calib_dl,
#                 val_dataloaders=calib_dl,
#                 ckpt_path=None
#             )
#
#             # then compute the score quantile
#             with torch.no_grad():
#                 self.pt.test(
#                     self.trainable_model,
#                     dataloaders=calib_dl
#                 )
#                 scores, labels = self.trainable_model.latest_test_results
#                 label_scores = torch.gather(scores, 1, labels.unsqueeze(1)).squeeze()
#                 # TODO quantile correction 1/n+1?
#                 quantile = self.trainable_model.score_fn.compute_quantile(label_scores, self.trainable_model.alpha)
#
#         return quantile
