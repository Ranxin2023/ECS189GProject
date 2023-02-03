"""
Concrete ResultModule class for a specific experiment ResultModule output
"""
from abc import ABC

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ECS189G_Winter_2022_Source_Code_Template.code.base_class.result import BaseResult
import pickle


class ResultSaver(BaseResult, ABC):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        print('saving results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count),
                 'wb')
        pickle.dump(self.data, f)
        f.close()
