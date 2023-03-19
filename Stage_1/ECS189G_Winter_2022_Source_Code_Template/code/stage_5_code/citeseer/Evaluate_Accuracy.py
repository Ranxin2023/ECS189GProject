"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import ignite as ignite
import numpy as np

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.evaluate import BaseEvaluate
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import torch
from collections import OrderedDict

import torch
from torch import nn, optim
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall, ClassificationReport
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar



class EvaluateAccuracy(BaseEvaluate):
    data = None

    def eval_step(engine, batch):
        return batch

    def evaluate(self):
        print('evaluating performance...')
        # print(np.array(torch.tensor(self.data['true_y'], device = 'cpu')))
        return classification_report(self.data['true_y'], self.data['pred_y'])
        # TODO

    def classification_report2(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return np.round(np.array(correct / len(labels)), 2)

    def class2(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        return classification_report(preds, labels)


    # def data_reporter(self):
    #
    #     print(f"Accuracy Score of the classifier is: {accuracy_score(self.data['true_y'].cpu().numpy(), self.data['pred_y'].cpu().numpy())}")
    #     print(f"Precision Score of the classifier is: {precision_score(self.data['true_y'].cpu().numpy(), self.data['pred_y'].cpu().numpy(), average = 'weighted')}")
    #     print(f"Recall Score of the classifier is: {recall_score(self.data['true_y'].cpu().numpy(), self.data['pred_y'].cpu().numpy(), average='weighted')}")
    #     print(f"f1 Score of the classifier is: {f1_score(self.data['true_y'].cpu().numpy(), self.data['pred_y'].cpu().numpy(), average='weighted' )}")
    #
