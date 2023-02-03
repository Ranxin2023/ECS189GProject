"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Evaluate_Accuracy import EvaluateAccuracy
import torch
from torch import nn
import numpy as np



class MethodMLP(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 1000
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3        # TODO: change learning rate to increase evaluate accuracy

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 512)       # TODO: change to (784, 你看着办）
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 256)       # TODO：change to (你看着办， 你继续看着办)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(256, 64)

        self.fc_layer_4 = nn.Linear(64, 10)
        self.activation_func_3 = nn.ReLU()
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_4 = nn.Softmax(dim=1)  # TODO: try different activation functions to see which one fits best

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        # create more layers in initializations: fc_layer_3
        """Forward propagation"""
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        h_next = self.activation_func_2(self.fc_layer_2(h))
        h_nn  = self.activation_func_3(self.fc_layer_3(h_next))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_4(self.fc_layer_4(h_nn))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, x, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(x)))   # dataloader already did np.array
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
                #self.loss_set.append(train_loss.item())

    def test(self, x):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(x)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['x'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['x'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            