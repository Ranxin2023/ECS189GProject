"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod


import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.pygcn import GraphConvolution
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.Evaluate_Accuracy import EvaluateAccuracy
import torch.nn.functional as F


class MethodGCN(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3  # TODO: change learning rate to increase evaluate accuracy

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.gc1 = GraphConvolution(3703, 32)
        self.gc2 = GraphConvolution(32, 18)
        self.gc3 = GraphConvolution(18, 6)


    def forward(self, x, adj):
        # create more layers in initializations: fc_layer_3
        """Forward propagation"""
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.5)
        x = self.gc2(x, adj)
        x = F.dropout(x, 0.5)
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    total_train_step = 0

    def train(self):

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # TODO: try different optimizer to see results: SGD, etc
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')
        for epoch in range(self.max_epoch):
            global total_train_step
            total_train_step = 0
            optimizer.zero_grad()
            output = self.forward(self.data['graph']['X'], self.data['graph']['utility']['A'])
            train_loss = loss_function(output[self.data['train_test_val']['idx_train']], self.data['graph']['y'][self.data['train_test_val']['idx_train']])
            train_loss.backward()
            optimizer.step()
            if total_train_step % 200 == 0:
                print('Epoch:', epoch, 'Total train step', total_train_step, 'Loss:', train_loss.item(), 'Accuracy', accuracy_evaluator.classification_report2(output[self.data['train_test_val']['idx_train']], self.data['graph']['y'][self.data['train_test_val']['idx_train']]))
                total_train_step = total_train_step + 1
                self.loss_set.append(train_loss.item())

    def test(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            output = self.forward(self.data['graph']['X'], self.data['graph']['utility']['A'])
            train_loss = loss_function(output[self.data['train_test_val']['idx_test']],self.data['graph']['y'][self.data['train_test_val']['idx_test']])
            print('Loss:', train_loss.item())


        return output[self.data['train_test_val']['idx_test']], self.data['graph']['y'][self.data['train_test_val']['idx_test']]

    def run(self):
        print('method running...')
        print('--start training...')

        self.train()
        print('--start testing...')

        output, labels = self.test()
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')
        print('Accuracy', accuracy_evaluator.classification_report2(output, labels))
        print(accuracy_evaluator.class2(output, labels))


