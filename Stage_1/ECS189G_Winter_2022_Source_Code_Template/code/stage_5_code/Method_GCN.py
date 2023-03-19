"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.Evaluate_Accuracy import EvaluateAccuracy
import torch
import math
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    GraphConvolution reference https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

    """
    GraphConvolution reference https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """
class MethodGCN(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 2000
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.0004

    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(1433, 60)
        self.gc2 = GraphConvolution(60, 30)
        self.gc3 = GraphConvolution(30, 7)
        self.dropout = 0.15

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    # def train(self, x, y):
    def train(self, data, adj, features, labels, idx_train, idx_val, idx_test, nfeat, nclass):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            output = self.forward(features, adj)
            train_loss = loss_function(output[idx_train], labels[idx_train])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                pred_y = output[idx_train].max(1)[1].type_as(labels[idx_train])
                accuracy_evaluator.data = {'true_y': labels[idx_train], 'pred_y': pred_y}
                self.loss_set.append(train_loss.item())
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, features, adj):
        y_pred = self.forward(features, adj)
        return y_pred

    def run(self):
        print('method running...')
        print('--start training...')

        adj = self.data['graph']['utility']['A']
        features = self.data['graph']['X']
        labels = self.data['graph']['y']
        idx_train = self.data['train_test_val']['idx_train']
        idx_val = self.data['train_test_val']['idx_val']
        idx_test = self.data['train_test_val']['idx_test']

        nfeat = features.shape[1]
        nclass = labels.max().item() + 1

        self.train(self.data, adj, features, labels, idx_train, idx_val, idx_test, nfeat, nclass)
        print('--start testing...')
        predict = self.test(features, adj)
        pred_y = predict[idx_test].max(1)[1].type_as(labels[idx_test])
        return {'pred_y': pred_y, 'true_y': labels[idx_test]}
