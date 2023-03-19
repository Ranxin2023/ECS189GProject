from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.pubmed.Dataset_Loader import DatasetLoader
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.pubmed.Method_GCN import MethodGCN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.pubmed.Result_Saver import ResultSaver
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.pubmed.Setting_Train_Test_Split import SettingGCN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.pubmed.Evaluate_Accuracy import EvaluateAccuracy
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
    data_obj = DatasetLoader('stage5_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed'
    data_obj.dataset_name = 'pubmed'

    method_obj = MethodGCN('GCN', '')

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingGCN('Train Test Split', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)         # check inside base functions
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('GCN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    evaluate_obj.data_reporter()
    print(evaluate_obj.classification_report())
    print('************ Finish ************')
    # ------------------------------------------------------

    epoch = [i for i in range(1500) if i % 10 == 0]
    print (epoch)
    plt.plot(epoch, method_obj.loss_set, color='green', linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor='blue', markersize=12)

    plt.title("Loss Term Values vs Training Epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss Term Values")

    plt.show()