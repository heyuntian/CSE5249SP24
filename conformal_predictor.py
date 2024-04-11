from typing import Dict

from dgl.dataloading import MultiLayerFullNeighborSampler
import torch

from constants import ScoreType, Stage
# from data_module import DataModule
from scores import *
from transformations import *
from itertools import combinations
from statistics import mean

class ConformalPredictor:
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha # coverage req

    def C(self, x):
        """Generate a set/interval of values such that $P(y \in C(x)) \geq 1 - \alpha$"""
        raise NotImplementedError

    def calculate_coverage(self, prediction_sets, labels, nc_score_type):
        includes_true_label = prediction_sets.gather(1, labels.unsqueeze(1)).squeeze()
        empirical_coverage = includes_true_label.sum()/len(prediction_sets)
        # print(f"The empirical coverage with {nc_score_type} is: {empirical_coverage}")
        return empirical_coverage.item()

    def calculate_efficiency(self, prediction_sets, nc_score_type):
        empirical_efficiency = (prediction_sets.sum(dim=1)).sum().div(len(prediction_sets))
        # print(f"The empirical efficiency with {nc_score_type} is: {empirical_efficiency}")
        return empirical_efficiency.item()

    def calculate_fairness(self, prediction_sets, sens, labels, nc_score_type):
        results = {}
        num_sens_attr = sens.shape[1]
        for sens_col in range(num_sens_attr):
            sens_values = max(sens[:, sens_col]) + 1
            dict_sens = {sens_val: sens[:, sens_col] == sens_val for sens_val in range(sens_values)}
            # coverage
            includes_true_label = prediction_sets.gather(1, labels.unsqueeze(1)).squeeze()
            coverages = {sens_val: (includes_true_label[dict_sens[sens_val]].sum() / dict_sens[sens_val].sum()).item() for sens_val in range(sens_values)}
            # print(coverages)
            results[f'{nc_score_type}-CoverageDisparity-{sens_col}'] = mean(abs(coverages[x] - coverages[y]) for x, y in combinations(range(sens_values), 2))
            # efficiency
            efficiencies = {sens_val: prediction_sets[dict_sens[sens_val]].sum(dim=1).sum().div(dict_sens[sens_val].sum()).item() for sens_val in range(sens_values)}
            # print(efficiencies)
            results[f'{nc_score_type}-EfficiencyDisparity-{sens_col}'] = mean(abs(efficiencies[x] - efficiencies[y]) for x, y in combinations(range(sens_values), 2))
        return results



class ConformalClassifier(ConformalPredictor):
    def __init__(self, alpha, n_classes, **kwargs):
        super().__init__(alpha, **kwargs)
        self.n_classes = n_classes
    

class SplitConformalClassifier(ConformalClassifier):
    def __init__(self, alpha, n_classes, **kwargs):
        super().__init__(alpha, n_classes, **kwargs)
    
    def calibrate(self, **calib_data):
        """Calibrate the conformal Predictor"""
        raise NotImplementedError
    

# Accumulator for probability for computing 
class ProbabilityAccumulator:
    def __init__(self) -> None:
        pass


class ScoreSplitConformalClassifer(SplitConformalClassifier):
    """A score based split conformal classifier"""
    def __init__(self, alpha, n_classes,  split_dict,
                 nc_score_type: ScoreType = ScoreType.APS, **kwargs):
        super().__init__(alpha, n_classes, **kwargs)
        self.nc_score_type = nc_score_type
        self.split_dict: Dict[Stage, torch.BoolTensor] = split_dict
        self._qhat = None
        self._score_module = None
        self._transform_module = None
        self._cached_scores = None

    def _get_scores(self, probs: torch.Tensor, use_aps_epsilon: bool, **kwargs): 
        # calibration using score quantile
        # assuming that score is exchangeable, this should work
        if self.nc_score_type == ScoreType.APS:
            self._score_module = APSScore(use_aps_epsilon=use_aps_epsilon, **kwargs)
        elif self.nc_score_type == ScoreType.DAPS:
            self._score_module = APSScore(use_aps_epsilon=use_aps_epsilon, **kwargs)
            self._transform_module = DiffusionTransformation(**kwargs)
        else:
            raise NotImplementedError
        
        scores = self._score_module.pipe_compute(probs)
        
        if self._transform_module is not None:
            scores = self._transform_module.pipe_transform(scores)
        return scores

    def calibrate(self, probs: torch.Tensor, labels: torch.Tensor, use_aps_epsilon: bool, **kwargs):
        split_dict = self.split_dict
        # split_mask = split_dict[Stage.TRAIN]\
        #             | split_dict[Stage.VALIDATION]\
        #             | split_dict[Stage.CALIBRATION]\
        #             | split_dict[Stage.TEST] # TODO do we need test also?

        scores = self._cached_scores
        if scores is None:
            scores = self._get_scores(probs, use_aps_epsilon, **kwargs)
            self._cached_scores = scores
        assert self._score_module is not None

        # label_scores = scores.gather(1, labels.unsqueeze(1)).squeeze()
        # label_scores = label_scores[split_dict[Stage.CALIBRATION]]
        # self._qhat = self._score_module.compute_quantile(label_scores, self.alpha)

        calib_labels = labels[split_dict[Stage.CALIBRATION]]
        calib_scores = scores[split_dict[Stage.CALIBRATION]].gather(1, calib_labels.unsqueeze(1)).squeeze()
        self._qhat = self._score_module.compute_quantile(calib_scores, self.alpha)

        return self._qhat
    
    def run(self, probs: torch.Tensor, labels: torch.Tensor,
            use_aps_epsilon=True, calc_fair=(False, None), **kwargs):
        qhat = self.calibrate(probs, labels, use_aps_epsilon=use_aps_epsilon, **kwargs)
        assert self._cached_scores is not None

        split_dict = self.split_dict
        test_labels = labels[split_dict[Stage.TEST]]
        test_scores = self._cached_scores[split_dict[Stage.TEST]]
        prediction_sets = PredSetTransformation(qhat=qhat).pipe_transform(test_scores)

        results = dict()
        results[f'{self.nc_score_type}-efficiency'] = self.calculate_efficiency(prediction_sets, self.nc_score_type)
        results[f'{self.nc_score_type}-coverage'] = self.calculate_coverage(prediction_sets, test_labels, self.nc_score_type)
        if calc_fair[0]:
            sens_test = calc_fair[1][split_dict[Stage.TEST]]
            fair_results = self.calculate_fairness(prediction_sets, sens_test, test_labels, self.nc_score_type)
            results.update(fair_results)
        return results

# class ScoreMultiSplitConformalClassifier(ScoreSplitConformalClassifer):
#     def __init__(self, alpha, n_classes, split_dict, nc_score_type: ScoreType = ScoreType.CFGNN, **kwargs):
#         super().__init__(alpha, n_classes, split_dict, nc_score_type, **kwargs)
#
#     def get_dataloader(self, datamodule, nodes, total_num_layers):
#         sampler = MultiLayerFullNeighborSampler(total_num_layers)
#         return datamodule.custom_nodes_dataloader(nodes, len(nodes), sampler)
#
#     def calibrate_with_model(self, base_model_path: str, use_aps_epsilon: bool, datamodule: DataModule, total_num_layers, **kwargs):
#         split_dict = self.split_dict
#         if self.nc_score_type == ScoreType.CFGNN:
#             self._score_module = CFGNNScore(base_model_path, use_aps_epsilon=use_aps_epsilon,
#                                             num_classes=self.n_classes,
#                                             **kwargs)
#             calib_nodes = torch.nonzero(split_dict[Stage.CALIBRATION], as_tuple=True)[0]
#             calib_dl = self.get_dataloader(datamodule, calib_nodes, total_num_layers)
#             self._qhat = self._score_module.learn_params(calib_dl)
#         else:
#             raise NotImplementedError
#         return self._qhat
#
#     def run(self, base_model_path, datamodule: DataModule, total_num_layers: int, use_aps_epsilon=True, **kwargs):
#         split_dict = self.split_dict
#         qhat = self.calibrate_with_model(base_model_path, use_aps_epsilon=use_aps_epsilon, datamodule=datamodule, total_num_layers=total_num_layers, **kwargs)
#
#         test_nodes = torch.nonzero(split_dict[Stage.TEST], as_tuple=True)[0]
#         test_dl = self.get_dataloader(datamodule, test_nodes, total_num_layers)
#         test_scores, test_labels = self._score_module.pipe_compute(test_dl)
#         prediction_sets = PredSetTransformation(qhat=qhat).pipe_transform(test_scores)
#
#         return self.calculate_efficiency(prediction_sets, self.nc_score_type), self.calculate_coverage(prediction_sets, test_labels, self.nc_score_type)

