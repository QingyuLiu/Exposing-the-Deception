import numpy as np
import torch
from metrics.metric import Metric


class TestLossMetric(Metric):

    def __init__(self, train=False):
        self.main_metric_name = 'value'
        super().__init__(name='Loss', train=False)

    def compute_metric(self, value):
        """Computes the precision@k for the specified values of k"""
        metrics = dict()
        for key, value in value.items():
            metrics[key] = np.mean(value)
        return metrics
    def accumulate(self, loss: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        return {'value': loss.item()}
