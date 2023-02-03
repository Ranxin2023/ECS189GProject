"""
Concrete IO class for a specific dataset
"""
import numpy as np

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.dataset import BaseDataset
import pandas as pd

class DatasetLoader(BaseDataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_train_name = None
    dataset_source_file_test_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_train_name)
        df_test = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_test_name)

        x_train = df.iloc[:, 1:]    # get everything but the fist list
        x_train = np.array(x_train)
        y_train = df.iloc[:, 0]     # get the first list
        y_train = np.array(y_train)

        x_test = df_test.iloc[:, 1:]
        x_test = np.array(x_test)
        y_test = df_test.iloc[:, 0]
        y_test = np.array(y_test)

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}