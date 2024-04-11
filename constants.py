import os

from enum import Enum

WORKING_DIRECTORY = os.path.dirname(__file__)

DATASET_DIRECTORY = os.path.join(WORKING_DIRECTORY, "datasets")
OUTPUT_DIRECTORY = os.path.join(WORKING_DIRECTORY, "outputs")

CPU_AFF = "enable_cpu_affinity"


class Stage(str, Enum):
    UNLABELED = "unlabeled"
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"

    @property
    def mask_dstr(self):
        return {
            Stage.UNLABELED: "unlabeled_mask",
            Stage.TRAIN: "train_mask",
            Stage.VALIDATION: "val_mask",
            Stage.CALIBRATION: "calib_mask",
            Stage.TEST: "test_mask"
        }[self]


class ScoreType(str, Enum):
    APS = "aps"
    DAPS = "daps"
    CFGNN = "cfgnn"

CORA = "Cora"
CITESEER = "CiteSeer"
PUBMED = "PubMed"
COAUTHOR_CS = "Coauthor_CS"
COAUTHOR_PHYSICS = "Coauthor_Physics"
OGBN_PRODUCTS = "ogbn_products"

CLASSIFICATION_DATASETS = [CORA, CITESEER, PUBMED, COAUTHOR_CS, COAUTHOR_PHYSICS, OGBN_PRODUCTS]
CUSTOM_DATASETS = [CITESEER]