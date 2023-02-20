"""
Concrete IO class for a specific dataset
"""
import numpy as np
import pickle
from sklearn.utils import shuffle

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.config import *
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.dataset import BaseDataset
import pandas as pd

class DatasetLoader(BaseDataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')    #Finish Changing base on CNN dataset structure, line 24-34

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for pair in data['train']:
            x_train.append([pair['image']])
            y_train.append(pair['label'])
        for pair in data['test']:
            x_test.append([pair['image']])
            y_test.append(pair['label'])

        x_train, x_test = np.array(x_train[0:59968]), np.array(x_test[0:9984])
        x_train, x_test = self.data_normalization(x_train, x_test)

        x_train = self.generator_batch_size(x_train, 64)
        y_train = self.generator_batch_size(y_train[0:59968], 64)

        x_test = self.generator_batch_size(x_test, 64)
        y_test = self.generator_batch_size(y_test[0:9984], 64)

        x_test = np.array(x_test)
        f.close()


        x_train, y_train = shuffle(x_train, y_train)
        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


    def data_normalization(self, train, test):
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')

        train_norm = train_norm /255.0
        test_norm = test_norm / 255.0

        return train_norm, test_norm

    def generator_batch_size(self, data, batch_size):


        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

