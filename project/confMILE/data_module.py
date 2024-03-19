import os
import shutil

import dgl
from dgl.data import DGLDataset, CoraFullDataset, PubmedGraphDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import numpy as np
import pytorch_lightning as pl
import torch

from constants import *
# from data_utils import data_prep

class ClassificationDataset(DGLDataset):
    def __init__(self, name, train_frac=0.2, val_frac=0.1, calib_frac=0.35, graph: dgl.DGLGraph = None):
        self.dataset_path = os.path.join(DATASET_DIRECTORY, name)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.calib_frac = calib_frac
        self.graph = graph
        super().__init__(name=name)

    def process(self):
        if self.graph is None:
            edge_list = torch.load(os.path.join(self.dataset_path, "processed/edge_list.pt"))
            features = torch.load(os.path.join(self.dataset_path, "processed/features.pt"))
            labels = torch.load(os.path.join(self.dataset_path, "processed/labels.pt"))
            n_nodes = labels.shape[0]

            self.graph = dgl.graph((edge_list[0], edge_list[1]), num_nodes=n_nodes)
            self.graph = dgl.add_self_loop(self.graph)
        else:
            features = self.graph.ndata.pop("feat")
            labels = self.graph.ndata.pop("label")
            n_nodes = labels.shape[0]

        self.graph.ndata["features"] = features
        self.graph.ndata["labels"] = labels

        n_train = int(n_nodes * self.train_frac)
        n_val = int(n_nodes * self.val_frac)
        n_calib = int(n_nodes * self.calib_frac)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        calib_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        r_order = np.random.permutation(n_nodes) # Randomize order of nodes
        
        train_mask[r_order[:n_train]] = True
        val_mask[r_order[n_train : n_train + n_val]] = True
        calib_mask[r_order[n_train + n_val : n_train + n_val + n_calib]] = True
        test_mask[r_order[n_train + n_val + n_calib : ]] = True

        self.graph.ndata[Stage.TRAIN.mask_dstr] = train_mask
        self.graph.ndata[Stage.VALIDATION.mask_dstr] = val_mask
        self.graph.ndata[Stage.CALIBRATION.mask_dstr] = calib_mask
        self.graph.ndata[Stage.TEST.mask_dstr] = test_mask
    
    def get_mask_inds(self, mask_key: str):
        mask = torch.Tensor(self.graph.ndata[mask_key])
        return torch.nonzero(mask, as_tuple=True)[0]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        return self.graph


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        num_layers: int,
        seed: int = 0,
        batch_size: int = 256,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.name = name
        self.seed = seed
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.has_setup = False

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_features(self) -> int:
        features = torch.Tensor(self.graph.ndata["features"])
        return features.shape[1]


    @property
    def num_classes(self) -> int:
        labels = torch.Tensor(self.graph.ndata["labels"])
        return labels.unique().shape[0]

    # def prepare_data(self, reprocess=False) -> None:
    #     assert self.name is not None and self.name in CLASSIFICATION_DATASETS
    #
    #     if self.name in CUSTOM_DATASETS:
    #         dataset_path = os.path.join(DATASET_DIRECTORY, self.name)
    #         processed_path = os.path.join(dataset_path, "processed")
    #         if os.path.exists(processed_path) and reprocess:
    #             shutil.rmtree(processed_path)
    #
    #         if not os.path.exists(os.path.join(dataset_path, "processed")):
    #             os.mkdir(os.path.join(dataset_path, "processed"))
    #             data_prep(self.name, dataset_path)
    #     else:
    #         if self.name == CORA:
    #             CoraFullDataset(DATASET_DIRECTORY)
    #         elif self.name == PUBMED:
    #             PubmedGraphDataset(DATASET_DIRECTORY)
    #         elif self.name == COAUTHOR_CS:
    #             CoauthorCSDataset(DATASET_DIRECTORY)
    #         elif self.name == COAUTHOR_PHYSICS:
    #             CoauthorPhysicsDataset(DATASET_DIRECTORY)


    def setup(self, stage: str) -> None:
        if not self.has_setup:
            self.sampler = MultiLayerFullNeighborSampler(self.num_layers)

            # Use DGL Dataset when possible
            dataset = None
            if self.name == CORA:
                dataset = CoraFullDataset(DATASET_DIRECTORY)
            elif self.name == PUBMED:
                dataset = PubmedGraphDataset(DATASET_DIRECTORY)
            elif self.name == COAUTHOR_CS:
                dataset = CoauthorCSDataset(DATASET_DIRECTORY)
            elif self.name == COAUTHOR_PHYSICS:
                dataset = CoauthorPhysicsDataset(DATASET_DIRECTORY)

            if self.name in CLASSIFICATION_DATASETS:
                if dataset is None:
                    dataset = ClassificationDataset(self.name)
                else:
                    dataset = ClassificationDataset(self.name, graph=dataset[0])
            else:
                raise NotImplementedError

            self.graph = dataset[0]
            self.split_dict = {
                stage: dataset.get_mask_inds(stage.mask_dstr)
                for stage in Stage
            }

            self.has_setup = True

    def train_dataloader(self):
        return DataLoader(
            self.graph,
            self.split_dict[Stage.TRAIN],
            self.sampler,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.graph,
            self.split_dict[Stage.VALIDATION],
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # TODO: fix this for non DAPS methods
        return DataLoader(
                self.graph,
                #torch.hstack((self.split_dict[Stage.TEST], self.split_dict[Stage.CALIBRATION])),
                self.split_dict[Stage.TEST],
                self.sampler,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers
            )

    def all_dataloader(self):
        return DataLoader(
            self.graph,
            torch.arange(self.num_nodes),
            self.sampler,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )

    def custom_nodes_dataloader(self, nodes, batch_size:int, sampler=None):
        if sampler is None:
            sampler = self.sampler
        return DataLoader(
            self.graph,
            nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
