from abc import ABC

import torch

class Transformation(ABC):
    def __init__(self, **kwargs):
        self.defined_args = kwargs

    def pipe_transform(self, x):
        return self.transform(x, **self.defined_args)
    
    def transform(self, x, **kwargs):
        return x

class PredSetTransformation(Transformation):
    def transform(self, x, **kwargs):
        qhat = kwargs.get("qhat")
        return x <= qhat

class DiffusionTransformation(Transformation):
    def transform(self, x, **kwargs):
        diffusion_param = kwargs.get("diffusion_param", 0)
        edge_index = kwargs.get("edge_index")
        num_nodes = kwargs.get("num_nodes")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = x.to(device)
        A = torch.sparse.FloatTensor(
            edge_index.to(device),
            torch.ones(edge_index.shape[1]).to(device),
            (num_nodes, num_nodes)
        )
        # TODO: More efficient, possibly batched computation
        # Also TODO: Choose diffusion param based on tuning
        degs = torch.matmul(A, torch.ones((A.shape[0])).to(device))

        return ((1 - diffusion_param) * x + diffusion_param * (1 / (degs + 1e-10))[:, None] * torch.linalg.matmul(A, x)).cpu()
        