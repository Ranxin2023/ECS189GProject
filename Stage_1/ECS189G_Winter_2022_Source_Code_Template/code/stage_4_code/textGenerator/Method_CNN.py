"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.method import BaseMethod
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.MNIST.Evaluate_Accuracy import EvaluateAccuracy
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.CIFAR.config import *
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.textGenerator.Dataset_Loader import DatasetLoader
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
class MethodCNN(BaseMethod, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3        # TODO: change learning rate to increase evaluate accuracy

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(num_embeddings= 4670, embedding_dim= 128)

        self.lstm = nn.LSTM(input_size= 128, hidden_size= 128, num_layers=3, dropout=0.5)
        self.fc = nn.Linear(128, 4670)

    def forward(self, x, pre_state):
        # create more layers in initializations: fc_layer_3
        """Forward propagation"""
        embed = self.embedding(x)
        output, state  = self.lstm(embed, pre_state)
        log = self.fc(output)
        return log, state



    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    total_train_step = 0
    def train(self, X):

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
            state_h, state_c = torch.zeros(3, 5, 128), torch.zeros(3, 5, 128)
            # for imgs, targets in tqdm(train_loader, test_loader, total = len(train_loader)):
            #     imgs, targets = imgs.to(device), targets.to(device)
            #     optimizer.zero_grad()
            for data in tqdm(X, total = len(X)):
                optimizer.zero_grad()
                x = torch.tensor(data[:-1])

            # print('targets:{}'.format(targets))
                y = torch.LongTensor(data[1:])
                # print(targets)
                y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))

                # print(output)
                train_loss = loss_function(y_pred.transpose(1, 2), y)
                state_h, state_c = state_h.detach(), state_c.detach()
                train_loss.backward()
                optimizer.step()
                if total_train_step%200 == 0:
                    print('Epoch:', epoch, 'Total train step', total_train_step,'Loss:', train_loss.item())
                    total_train_step = total_train_step + 1

                    self.loss_set.append(train_loss.item())


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



    # def test(self, test_loader, Y):
    #     correct = 0
    #     total = 0
    #     predict = []
    #     with torch.no_grad():
    #         # for imgs, targets in test_loader:
    #         #     imgs, targets = imgs.to(device), targets.to(device)
    #         for test_loader, y in zip(test_loader, Y):
    #             # test_loader = test_loader.reshape(64, 1, 28, 28)
    #             imgs, targets =  torch.FloatTensor(test_loader).to(device), torch.LongTensor(y).to(device)
    #
    #             output = self.forward(imgs)
    #             _, predicted = torch.max(output.data, 1)
    #             total += targets.size(0)
    #             correct += (predicted == targets).sum().item()
    #             predict.append(predicted)
    #     print('Test Accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100.*correct/total))

            # y_pred = self.forward(torch.FloatTensor(test_loader)).to(device)
        # return predict

    def predict(self, text, next_words = 5):
        words = text.split(' ')
        state_h, state_c = torch.zeros(3, len(words), 128), torch.zeros(3, len(words), 128)

        for i in range(0, next_words):
            x = torch.tensor([[self.data['dictionary'][w] for w in words[i:]]])
            y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))
            last_words = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_words, dim = 0).detach().numpy()
            word_index = np.random.choice(len(last_words), p=p)

            words.append(list(self.data['dictionary'].keys())[list(self.data['dictionary'].values()).index(word_index)])

        return words
        # return y_pred.max(1)[1]
        # do the testing, and result the result
        #y_pred = self.forward(torch.FloatTensor(np.array(x)))               # TODO: Double check
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        #return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')

        self.train(self.data['train']['x'])
        print('--start testing...')
        text = 'why did the chicken kill'
        pred_y = self.predict(text)

        return pred_y