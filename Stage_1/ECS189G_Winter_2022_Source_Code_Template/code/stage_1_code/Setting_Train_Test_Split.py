"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting
from sklearn.model_selection import train_test_split
import numpy as np


class SettingTrainTestSplit(BaseSetting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        x_train, x_test, y_train, y_test = train_test_split(loaded_data['x'], loaded_data['y'], test_size=0.33)

        # run MethodModule
        self.method.data = {'train': {'x': x_train, 'y': y_train}, 'test': {'x': x_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        