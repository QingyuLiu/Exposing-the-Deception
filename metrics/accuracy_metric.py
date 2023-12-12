import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='Accuracy', train=False)

    def accumulate(self, values):
        """Computes the precision@k for the specified values of k"""
        outputs=values[0]
        labels=values[1]
        max_k = max(self.top_k)
        batch_size = labels.shape[0]

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = dict()
        res['correct']=torch.sum(correct).item()
        res['batch_size']=batch_size
        # for k in self.top_k:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        return res
    def compute_metric(self,values):
        correct = sum(values['correct'])
        batch_size = sum(values['batch_size'])
        res = dict()
        res['value'] = correct*(100.0 / batch_size)
        return res
    # def accumulate_acc(outputs, labels, top_k = (1,)):
    #     max_k = max(top_k)
    #     _, pred = outputs.topk(max_k, 1, True, True)
    #     pred = pred.t()
    #     correct = pred.eq(labels.view(1, -1).expand_as(pred))
    #     return torch.sum(correct).item()
    #
