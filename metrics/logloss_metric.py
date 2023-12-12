import torch
from metrics.metric import Metric
from sklearn.metrics import log_loss
from scipy.special import softmax
import numpy as np
from tkinter import _flatten
class LOGLOSSMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='LogLoss', train=False)

    def compute_metric(self, values):
        """Computes the precision@k for the specified values of k"""
        predicted = np.concatenate(values['prediction'])
        logloss=log_loss(list(_flatten(values['labels'])),predicted)
        return {'value': logloss}

    def accumulate(self, values):
        out = values[0].cpu().detach()
        labels = values[1]
        res = dict()
        res['labels'] = labels.cpu().detach().tolist()
        res['prediction'] = softmax(np.array(out.data.tolist()), axis=1)[:,1]
        return res