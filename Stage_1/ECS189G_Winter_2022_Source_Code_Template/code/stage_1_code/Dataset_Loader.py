"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.dataset import BaseDataset


class DatasetLoader(BaseDataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        x = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(' ')]
            x.append(elements[:-1])
            y.append(elements[-1])
        f.close()
        return {'x': x, 'y': y}
