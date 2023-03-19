"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Evaluate_Accuracy import EvaluateAccuracy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class MethodRNN(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.0006

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)


        # *******************RNN*******************
        self.rnn1 = nn.RNN(200, 150, 10)
        self.fc1 = nn.Linear(150, 64)
        self.fc2 = nn.Linear(64, 32)
        self.activation_func_1 = nn.ReLU()
        self.activation_func_2 = nn.Sigmoid()
        self.dropout = nn.Dropout(0.35)

        '''
        # *******************LSTM*******************
        self.lstm = nn.LSTM(200, 100, 8)
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.activation_func_1 = nn.ReLU()
        self.activation_func_2 = nn.Sigmoid()
        '''


    def forward(self, x):

        # *******************RNN*******************
        hidden_layer = torch.zeros(10, 100, 150)
        output, hidden_layer_2 = self.rnn1(x, hidden_layer)
        # Get the hidden layers by using [:, -1, :]
        # Reference: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
        h1 = self.fc1(output[:, -1, :])
        y_pred = self.activation_func_2(h1)

        '''
        # *******************LSTM*******************
        print(x.size())
        hidden_layer = Variable(torch.zeros(8, 100, 100))
        hidden_cell = Variable(torch.zeros(8, 100, 100))
        output, state = self.lstm(x, (hidden_layer, hidden_cell))
        h1 = self.fc1(output[:, -1, :])
        h2 = self.activation_func_1(h1)
        h3 = self.fc2(h2)
        y_pred = self.activation_func_2(h3)
        '''
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
            y_true = torch.LongTensor(np.array(y))
            y_pred = self.forward(torch.FloatTensor(np.array(x)))
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                self.loss_set.append(train_loss.item())
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
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
            