from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from metrics.auc_metric import AUCMetric
from metrics.logloss_metric import LOGLOSSMetric
import torch
from torch.utils.tensorboard import SummaryWriter
import logging

class plt_tensorboard():
    def __init__(self, args):
        wr = SummaryWriter(log_dir=f'runs/{args.name}')
        self.tb_writer = wr

        self.metrics = [AccuracyMetric(), TestLossMetric(),AUCMetric(),LOGLOSSMetric()]
    def accumulate_metrics(self,outputs, labels,loss):
        self.metrics[0].accumulate_on_batch([outputs, labels])
        self.metrics[1].accumulate_on_batch(loss)
        self.metrics[2].accumulate_on_batch([outputs, labels])
        self.metrics[3].accumulate_on_batch([outputs, labels])

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()


    def report_metrics(self,step, tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        try:
            print(f"AUC: {self.metrics[2].get_value()['value']}, "
                  f"ACC: {self.metrics[0].get_value()['value']}, "
                  f"LogLoss: {self.metrics[3].get_value()['value']}")
        except:
            print("error")
        return self.metrics[2].get_value()['value']


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def create_table(params):
    data = "| name | value | \n |-----|-----|"
    params=params.__dict__
    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data