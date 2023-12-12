import torch
from metrics.metric import Metric
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.special import softmax
from tkinter import _flatten
# from torch_metrics import AUC
class AUCMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='AUC', train=False)

    def compute_metric(self, values):
        """Computes the precision@k for the specified values of k"""
        predicted = np.concatenate(values['prediction'])
        auc = roc_auc_score(list(_flatten(values['labels'])), predicted)
        return {'value': auc}

    def accumulate(self, values):
        out = values[0].cpu().detach()
        labels = values[1]

        res = dict()
        res['labels'] = labels.cpu().detach().tolist()
        res['prediction'] = softmax(np.array(out.data.tolist()), axis=1)[:,1]
        return res

