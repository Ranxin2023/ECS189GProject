"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.evaluate import BaseEvaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class EvaluateAccuracy(BaseEvaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])



    def classification_report(self):

        print("Reporting")
        return classification_report(self.data['true_y'], self.data['pred_y'])
