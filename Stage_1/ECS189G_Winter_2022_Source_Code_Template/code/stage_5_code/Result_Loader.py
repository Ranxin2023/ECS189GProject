"""
Concrete ResultModule class for a specific experiment ResultModule output
"""
from abc import ABC

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.result import BaseResult
import pickle


class ResultLoader(BaseResult, ABC):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        print('loading results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count),
                 'rb')
        self.data = pickle.load(f)
        f.close()