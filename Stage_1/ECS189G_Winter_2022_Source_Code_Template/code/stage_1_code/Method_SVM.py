"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from sklearn import svm


class MethodSVM(BaseMethod):
    c = None
    data = None
    
    def train(self, x, y):
        # check here for the svm.SVC doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        model = svm.SVC(C=self.c)
        # model variable learning has been built-in by the fit() function
        # check here for the svm.fit doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
        model.fit(x, y)
        return model
    
    def test(self, model, x):
        # check here for the svm.predict doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
        return model.predict(x)
    
    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['x'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['x'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            