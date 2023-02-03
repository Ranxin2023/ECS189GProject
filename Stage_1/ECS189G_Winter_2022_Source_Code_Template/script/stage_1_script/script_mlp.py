from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Dataset_Loader import DatasetLoader
from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Method_MLP import MethodMLP
from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Result_Saver import ResultSaver
from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Setting_KFold_CV import SettingKFoldCV
from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Setting_Train_Test_Split import SettingTrainTestSplit
from ECS189G_Winter_2022_Source_Code_Template.code.stage_1_code.Evaluate_Accuracy import EvaluateAccuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----


if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = DatasetLoader('toy', '')
    data_obj.dataset_source_folder_path = '../../data/stage_1_data/'
    data_obj.dataset_source_file_name = 'toy_data_file.txt'

    method_obj = MethodMLP('multi-layer perceptron', '')

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_1_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingKFoldCV('k fold cross validation', '')
    # setting_obj = SettingTrainTestSplit('Train Test Split IDK what is this','')
    # in_Test_Split('train test split', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    