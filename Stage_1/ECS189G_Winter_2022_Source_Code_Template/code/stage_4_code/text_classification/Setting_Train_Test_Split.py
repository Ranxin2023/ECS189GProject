"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting
from sklearn.model_selection import train_test_split
import numpy as np


class SettingRNN(BaseSetting):
    fold = 5
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()       # loaded_data -> it is a dictionary

        # run MethodModule
        self.method.data = {'train': {'x': loaded_data['x_train'], 'y': loaded_data['y_train']},
                            'test': {'x': loaded_data['x_test'], 'y': loaded_data['y_test']}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        print("*****************learned_result*****************")
        print(learned_result)
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        