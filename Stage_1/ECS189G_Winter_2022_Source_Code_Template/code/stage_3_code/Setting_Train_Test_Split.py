"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.config import *
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision
from torch.utils.data import DataLoader


class SettingCNN(BaseSetting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()       # loaded_data -> it is a dictionary

        # run MethodModule                      # FINISHED TODO: Changed base on dataset architecture
        self.method.data = {'train': {'x': loaded_data['x_train'], 'y': loaded_data['y_train']},
                           'test': {'x': loaded_data['x_test'], 'y': loaded_data['y_test']}}
        # MNIST has given a public MNIST dataloader, but here we constructed our own
        # iterator = DataLoader(dataset=self.method.data['train'], batch_size = 128, shuffle = True)
        # train_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
        #                                                      transform=torchvision.transforms.Compose([
        #                                                          torchvision.transforms.ToTensor(),
        #                                                          torchvision.transforms.Normalize(
        #                                                              (0.1307,), (0.3081,))
        #                                                      ])),
        #                           batch_size=128, shuffle=True)
        # test_loader = DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
        #                                                     transform=torchvision.transforms.Compose([
        #                                                         torchvision.transforms.ToTensor(),
        #                                                         torchvision.transforms.Normalize(
        #                                                             (0.1307,), (0.3081,))
        #                                                     ])),
        #                          batch_size=128, shuffle=True)
        #
        # learned_result = self.method.run(train_loader, test_loader)    # FINISHED TODO: Check out run in Method CNN\

        learned_result = self.method.run()
        # print('setting 47:{}'.format(learned_result))
        print(learned_result)
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
        print('.................')
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

        