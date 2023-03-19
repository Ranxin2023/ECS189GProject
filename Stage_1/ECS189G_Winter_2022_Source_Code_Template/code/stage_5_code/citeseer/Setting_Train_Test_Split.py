"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.setting import BaseSetting



class SettingGCN(BaseSetting):
    fold = 5

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()  # loaded_data -> it is a dictionary

        # run MethodModule                      # FINISHED TODO: Changed base on dataset architecture
        self.method.data = {'train_test_val': loaded_data['train_test_val'], 'graph': loaded_data['graph']}


        learned_result = self.method.run()
        # print('setting 47:{}'.format(learned_result))

        # save raw ResultModule
        self.result.data = learned_result

        self.result.save()
        print('.................')
        self.evaluate.data = learned_result

        # return self.evaluate.evaluate(), None

