"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting
from sklearn.model_selection import KFold
import numpy as np


class SettingKFoldCV(BaseSetting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(loaded_data['x']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            x_train, x_test = np.array(loaded_data['x'])[train_index], np.array(loaded_data['x'])[test_index]
            y_train, y_test = np.array(loaded_data['y'])[train_index], np.array(loaded_data['y'])[test_index]
        
            # run MethodModule
            self.method.data = {'train': {'x': x_train, 'y': y_train}, 'test': {'x': x_test, 'y': y_test}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        