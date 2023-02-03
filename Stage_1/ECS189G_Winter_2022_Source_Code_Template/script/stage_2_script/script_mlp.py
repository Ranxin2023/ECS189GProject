from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Dataset_Loader import DatasetLoader
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Method_MLP import MethodMLP
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Result_Saver import ResultSaver
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Setting_Train_Test_Split import SettingMLP
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_2_code.Evaluate_Accuracy import EvaluateAccuracy
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
    data_obj = DatasetLoader('stage2_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_train_name = 'train.csv'
    data_obj.dataset_source_file_test_name = 'test.csv'

    method_obj = MethodMLP('multi-layer perceptron', '')

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingMLP('Train Test Split IDK what is this', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)         # check inside base functions
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    #print(method_obj.loss_set)


    # TODO: 绘制learning curve plots
    plt.plot([100,200,300,400,500,600,700,800,900,1000], method_obj.loss_set, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)

    plt.title("Loss Term Values vs Training Epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss Term Values")

    plt.show()



