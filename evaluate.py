from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, \
    roc_auc_score
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

convert_dict = {'unlabeled_mask': -1,
                'train_mask': 0,
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
        if 'efficiency' in metric_name:
            res += f"{metric_name}:\t" + "{0:.5f}".format(entry[metric_name]) + f" / {graph.label_range}" + "\n"
        else:
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
    results_aps = cp.run(probs, labels, use_aps_epsilon=ctrl.use_aps_epsilon, calc_fair=(ctrl.fair_eval, graph.sens))
    results.update(results_aps)

    # DAPS
    def get_edge_list(graph):
        src_ids = []
        dst_ids = []
        for i in range(graph.node_num):
            for j in range(graph.adj_idx[i], graph.adj_idx[i + 1]):
                src_ids.append(i)
                dst_ids.append(graph.adj_list[j])
        return torch.stack((torch.tensor(src_ids), torch.tensor(dst_ids)), dim=0)

    cp = ScoreSplitConformalClassifer(
        alpha=ctrl.alpha,
        n_classes=n_classes,
        split_dict=split_dict,
        nc_score_type=ScoreType.DAPS,
    )
    results_daps = cp.run(
        probs, labels,
        use_aps_epsilon=ctrl.use_aps_epsilon,
        edge_index=get_edge_list(graph),
        num_nodes=graph.node_num,
        diffusion_param=ctrl.diffusion_param,
        calc_fair=(ctrl.fair_eval, graph.sens)
    )
    results.update(results_daps)

    # Singleton
    results_single = predict_max_label(probs, split_dict, labels, ['auroc', 'accuracy', 'micro-f1', 'macro-f1'],
                                       group_str=Stage.TEST, calc_fair=(ctrl.fair_eval, graph.sens))
    # predict_max_label(probs, split_dict, labels, ['auroc', 'accuracy', 'micro-f1', 'macro-f1'], group_str=Stage.TRAIN)
    results.update(results_single)

    return results


def predict_max_label(probs, split_dict, labels, uti_metrics, group_str=Stage.TEST, calc_fair=(False, None)):
    """
    Metrics related to the class with maximum probability (normal classification).
    :param entry:
    :param probs:
    :param split_dict:
    :param labels:
    :param uti_metrics:
    :param node_group: 0-3 for train, valid, calib, test nodes
    :return:
    """
    results = dict()
    if group_str not in split_dict:
        raise KeyError(f'{group_str} not in split_dict. Valid options: {list(split_dict)}.')
    num_label_values = max(labels) + 1
    test_nodes = split_dict[group_str]
    y_true = labels[test_nodes]
    y_pred_prob = probs[test_nodes]
    y_pred = np.argmax(y_pred_prob, axis=1).flatten()
    for metric_name in uti_metrics:
        if metric_name == 'auroc':
            results[group_str + '|' + metric_name] = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
        elif metric_name[-3:] == '-f1':
            results[group_str + '|' + metric_name] = f1_score(y_true, y_pred, average=metric_name[:-3])
        elif metric_name in ['accuracy', 'acc']:
            results[group_str + '|' + metric_name] = accuracy_score(y_true, y_pred)
        else:
            raise NotImplementedError(f'Metric {metric_name} is not implemented.')

    if calc_fair[0]:
        sens = calc_fair[1][split_dict[Stage.TEST]]
        num_sens_attr = sens.shape[1]
        for sens_col in range(num_sens_attr):
            num_sens_values = max(sens[:, sens_col]) + 1
            dict_sens = {sens_val: sens[:, sens_col] == sens_val for sens_val in range(num_sens_values)}
            results[f'{group_str}|dp-{sens_col}'] = float(Singleton_DemographicParity(num_sens_values, dict_sens, y_pred,
                                                                     num_label_values))
            results[f'{group_str}|eo-{sens_col}'] = float(Singleton_EqualOpportunity(num_sens_values, dict_sens, y_true, y_pred,
                                                                    num_label_values))

    return results


def Singleton_DemographicParity(num_sens_values, dict_sens, y_pred, num_label_values):
    def Singleton_DemographicParity_Label(val):
        return np.std([(y_pred[dict_sens[sens_val]] == val).sum() / dict_sens[sens_val].sum() for sens_val in
                       range(num_sens_values)])

    return np.mean([Singleton_DemographicParity_Label(label_value) for label_value in range(1, num_label_values)])


def Singleton_EqualOpportunity(num_sens_values, dict_sens, y_true, y_pred, num_label_values):
    def Singleton_EqualOpportunity_Label(val):
        return np.std([((y_true[dict_sens[sens_val]] == val) & (y_pred[dict_sens[sens_val]] == val)).sum() / (
                    y_true[dict_sens[sens_val]] == val).sum()
                       for sens_val in range(num_sens_values)])

    return np.mean([Singleton_EqualOpportunity_Label(label_value) for label_value in range(1, num_label_values)])
