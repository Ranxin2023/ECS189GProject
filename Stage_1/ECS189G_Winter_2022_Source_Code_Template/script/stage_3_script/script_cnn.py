from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Dataset_Loader import DatasetLoader
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Method_CNN import MethodCNN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Result_Saver import ResultSaver
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Setting_Train_Test_Split import SettingCNN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_3_code.Evaluate_Accuracy import EvaluateAccuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script ----


if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('stage3_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'                # FINISHED TODO: change the dataset folder path
    # data_obj.dataset_source_file_name = 'MNIST'
    data_obj.dataset_source_file_name = 'ORL'                                       # FINISHED TODO: train and test are in the same file, change the dataset folder name

    method_obj = MethodCNN('multi-layer perceptron', '')                            # FINISHED TODO: Change Method to CNN

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'  # FINISHED TODO: Change result destination folder path
    result_obj.result_destination_file_name = 'prediction_result'                   # FINISHED TODO: Change result destination file name

    setting_obj = SettingCNN('Train Test Split', '')               # FINISHED TODO: Change Setting MLP to CNN

    evaluate_obj = EvaluateAccuracy('accuracy', '')                                 # FINISHED TODO: Fix Evaluate Accuracy to fit with CNN
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)         # FINISHED TODO: check inside base functions
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()                # FINISHED TODO: Double check dataset architecture
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------

    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]