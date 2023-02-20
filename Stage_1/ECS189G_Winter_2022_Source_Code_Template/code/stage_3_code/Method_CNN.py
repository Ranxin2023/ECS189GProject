"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Evaluate_Accuracy import EvaluateAccuracy
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.config import *
import torch.nn.functional as F
class MethodCNN(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 5
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3        # TODO: change learning rate to increase evaluate accuracy

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # self.cv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))
        # self.mp_layer_1 = nn.MaxPool2d(2, 2)
        # self.cv_layer_2 = nn.Conv2d(32, 32, 5)
        # self.cv_layer_3 = nn.Conv2d(32, 64, 5)
        # self.fl_layer_1 = nn.Flatten()
        # self.fc_layer_1 = nn.Linear(3*3*64, 256)
        # self.activation_func_1 = nn.ReLU()
        # self.fc_layer_2 = nn.Linear(256, 10)
        # self.softmax = nn.Softmax(dim = 1)
        # self.drop_out = nn.Dropout(p=0.5)
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #
        # self.maxPooling = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(7744, 256)
        # self.fc2 = nn.Linear(256, 10)
        # self.drop = nn.Dropout(0.5)
        # self.relu = nn.ReLU()
        # self.activation_function = nn.Softmax(dim=1)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        # create more layers in initializations: fc_layer_3
        """Forward propagation"""
        # x = self.cv_layer_1(x)
        # x = self.activation_func_1(x)
        # x = self.cv_layer_2(x)
        # x = self.mp_layer_1(x)
        # x = self.activation_func_1(x)
        # x = self.drop_out(x)
        # x = self.cv_layer_3(x)
        # x = self.mp_layer_1(x)
        # x = self.activation_func_1(x)
        # x = self.drop_out(x)
        # x = self.fc_layer_1(x)
        # x = self.activation_func_1(x)
        # x = self.drop_out(x)
        # x = self.fc_layer_2(x)
        # y_pred = nn.Softmax(x, dim = 1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)





    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    total_train_step = 0
    def train(self, X, y):

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
         #TODO: try different optimizer to see results: SGD, etc
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        loss_function = nn.CrossEntropyLoss()

        # for training accuracy investigation purpose
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')
        for epoch in range(self.max_epoch):
            global total_train_step
            total_train_step = 0

            # for imgs, targets in tqdm(train_loader, test_loader, total = len(train_loader)):
            #     imgs, targets = imgs.to(device), targets.to(device)
            #     optimizer.zero_grad()
            for imgs, targets in tqdm(zip(X, y), total = len(y)):
                imgs = torch.FloatTensor(imgs)
            # print('targets:{}'.format(targets))
                targets = torch.LongTensor(targets)
                # print(targets)
                optimizer.zero_grad()
                output = self.forward(imgs)
                # print(output)
                train_loss = loss_function(output, targets)

                train_loss.backward()
                optimizer.step()
                if total_train_step%200 == 0:
                    accuracy_evaluator.data = {'true_y': targets, 'pred_y': output.max(1)[1]}
                    print('Epoch:', epoch, 'Total train step', total_train_step,
                            'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
                    total_train_step = total_train_step + 1
            # self.test(test_loader)
            # it will be an iterative gradient updating process
            # we don't do mini-batch, we use the whole input as one batch
            # you can try to split X and y into smaller-sized batches by yourself
            #for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                #y_pred = self.forward(torch.FloatTensor(np.array()))   # dataloader already did np.array
                # convert y to torch.tensor as well
                #y_true = torch.LongTensor(np.array(y))
                # calculate the training loss
                #train_loss = loss_function(y_pred, y_true)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                #optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                #train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                #optimizer.step()



    def test(self, test_loader, Y):
        correct = 0
        total = 0
        predict = []
        with torch.no_grad():
            # for imgs, targets in test_loader:
            #     imgs, targets = imgs.to(device), targets.to(device)
            for test_loader, y in zip(test_loader, Y):
                # test_loader = test_loader.reshape(64, 1, 28, 28)
                imgs, targets =  torch.FloatTensor(test_loader), torch.LongTensor(y)

                output = self.forward(imgs)
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                predict.append(predicted)
        print('Test Accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100.*correct/total))

            # y_pred = self.forward(torch.FloatTensor(test_loader)).to(device)
        return predict
        # return y_pred.max(1)[1]
        # do the testing, and result the result
        #y_pred = self.forward(torch.FloatTensor(np.array(x)))               # TODO: Double check
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        #return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['x'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['x'], self.data['test']['y'])
        # print('pred_y:{}'.format(pred_y))
        #
        # print('CNN 127:{}'.format(pred_y))
        print('.................................')

        print('abcd:{}'.format(self.data['test']['y']))
        # pred_y = np.array(pred_y[0].flatten())
        print('abcd:{}'.format(len(pred_y)))

        result = []
        for i in range(len(pred_y)):
            array = np.array(pred_y[i]).flatten().tolist()

            result.append(array)
        pred_y = np.array(result).flatten()

        self.data['test']['y'] = np.array(self.data['test']['y']).flatten()

        # print('data_type:{}'.format(pred_y[0]))

        # print('abcd:{}'.format(self.data['test']['y']))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            