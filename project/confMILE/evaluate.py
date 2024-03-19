from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import os
from typing import Sequence, Tuple, List
from scipy import stats



import torch
from constants import OUTPUT_DIRECTORY, ScoreType, Stage
from conformal_predictor import (
    ScoreSplitConformalClassifer,
    # ScoreMultiSplitConformalClassifier
)

convert_dict = {'train_mask': 0,
                    'val_mask': 1,
                    'calib_mask': 2,
                    'test_mask': 3}

def print_summary(ctrl, graph, entry):
    res = "************************************************************" + "\n"
    res += f"Dataset    :\t{ctrl.dataset} ({graph.node_num} nodes, {graph.edge_num} edges)\n"
    res += "Basic Embed:\t" + ctrl.basic_embed + "\n"
    if not ctrl.baseline:
        res += "Model:\tConfMILE\n"
        res += "Coarsen level:\t" + str(ctrl.coarsen_level) + "\n"
    for metric_name in entry:
        res += f"{metric_name}:\t" + "{0:.5f}".format(entry[metric_name]) + "\n"
    res += "Running time:\t" + "{0:.5f}".format(ctrl.embed_time) + " seconds" + "\n"
    res += "************************************************************" + "\n"
    print(res)

def conformal_evaluation(ctrl, probs, graph):
    probs = torch.tensor(probs)
    labels = torch.tensor(graph.labels)
    status = graph.status
    assert isinstance(probs, torch.Tensor) and isinstance(labels, torch.Tensor)
    n_classes = probs.shape[1]

    # create split_dict
    split_dict = {stage: torch.tensor(status == convert_dict[stage.mask_dstr])
        for stage in Stage
    }


    results = {}
    # APS
    cp = ScoreSplitConformalClassifer(
        alpha=ctrl.alpha,
        n_classes=n_classes,
        split_dict=split_dict,
        nc_score_type=ScoreType.APS,
    )
    efficiency, coverage = cp.run(probs, labels, use_aps_epsilon=ctrl.use_aps_epsilon)
    results[f'{ScoreType.APS}-efficiency'] = efficiency
    results[f'{ScoreType.APS}-coverage'] = coverage

    # DAPS
    def get_edge_list(graph):
        src_ids = []
        dst_ids = []
        for i in range(graph.node_num):
            for j in range(graph.adj_idx[i], graph.adj_idx[i+1]):
                src_ids.append(i)
                dst_ids.append(graph.adj_list[j])
        return torch.stack((torch.tensor(src_ids), torch.tensor(dst_ids)), dim=0)

    cp = ScoreSplitConformalClassifer(
        alpha=ctrl.alpha,
        n_classes=n_classes,
        split_dict=split_dict,
        nc_score_type=ScoreType.DAPS,
    )
    efficiency, coverage = cp.run(
        probs, labels,
        use_aps_epsilon=ctrl.use_aps_epsilon,
        edge_index=get_edge_list(graph),
        num_nodes=graph.node_num,
        diffusion_param=ctrl.diffusion_param
    )
    results[f'{ScoreType.DAPS}-efficiency'] = efficiency
    results[f'{ScoreType.DAPS}-coverage'] = coverage

    predict_max_label(results, probs, status, labels, ['auroc', 'accuracy', 'micro-f1', 'macro-f1'])
    # predict_max_label(results, probs, status, labels, ['auroc', 'accuracy', 'micro-f1', 'macro-f1'], group_str='train_mask')
    return results


def predict_max_label(entry, probs, status, labels, metrics, group_str='test_mask'):
    """
    Metrics related to the class with maximum probability (normal classification).
    :param entry:
    :param probs:
    :param status:
    :param labels:
    :param metrics:
    :param node_group: 0-3 for train, valid, calib, test nodes
    :return:
    """
    if group_str not in convert_dict:
        raise KeyError(f'{group_str} not in convert_dict. Valid options: {list(convert_dict)}.')
    test_nodes = status == convert_dict[group_str]
    y_true = labels[test_nodes]
    y_pred_prob = probs[test_nodes]
    y_pred = np.argmax(y_pred_prob, axis=1).flatten()
    for metric_name in metrics:
        if metric_name == 'auroc':
            entry[group_str + '|' + metric_name] = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
        elif metric_name[-3:] == '-f1':
            entry[group_str + '|' + metric_name] = f1_score(y_true, y_pred, average=metric_name[:-3])
        elif metric_name in ['accuracy', 'acc']:
            entry[group_str + '|' + metric_name] = accuracy_score(y_true, y_pred)
        else:
            raise NotImplementedError(f'Metric {metric_name} is not implemented.')