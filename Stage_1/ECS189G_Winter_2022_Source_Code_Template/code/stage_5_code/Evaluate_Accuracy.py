"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.evaluate import BaseEvaluate
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score


class EvaluateAccuracy(BaseEvaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])


    def classification_report(self):
        return classification_report(self.data['true_y'], self.data['pred_y'])

    def data_reporter(self):
        print(f"Accuracy Score of the classifier is: {accuracy_score(self.data['true_y'], self.data['pred_y'])}")
        print(
            f"Precision Score of the Classifier is：{precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')}"
        )
        print(
            f"Recall Score of the Classifier is：{recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')}"
        )
        print(
            f"f1 Score of the Classifier is：{f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')}"
        )