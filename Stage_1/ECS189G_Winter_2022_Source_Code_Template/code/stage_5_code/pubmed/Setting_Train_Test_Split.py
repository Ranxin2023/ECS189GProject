"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


class SettingGCN(BaseSetting):
    fold = 5
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()       # loaded_data -> it is a dictionary

        # run MethodModule

        self.method.data = {'graph': {'node': loaded_data['graph']['node'], 'edge': loaded_data['graph']['edge'],
                                      'X': loaded_data['graph']['X'], 'y': loaded_data['graph']['y'],
                                      'utility': {'A': loaded_data['graph']['utility']['A'],
                                                  'reverse_idx': loaded_data['graph']['utility']['reverse_idx']}},
                            'train_test_val': {'idx_train': loaded_data['train_test_val']['idx_train'],
                                               'idx_test': loaded_data['train_test_val']['idx_test'],
                                               'idx_val': loaded_data['train_test_val']['idx_val']}}

        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        print("*****************learned_result*****************")
        print(learned_result)
        self.result.save()
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        