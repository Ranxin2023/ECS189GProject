from matplotlib import pyplot as plt
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.Dataset_Loader_Node_Classification import Dataset_Loader
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.Method_GCN import MethodGCN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.textGenerator.Result_Saver import ResultSaver
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.Setting_Train_Test_Split import SettingGCN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_5_code.citeseer.Evaluate_Accuracy import EvaluateAccuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage5_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/citeseer/'                # FINISHED TODO: change the dataset folder path
    data_obj.dataset_source_file_name = 'data'                                     # FINISHED TODO: train and test are in the same file, change the dataset folder name


    method_obj = MethodGCN('multi-layer perceptron', '')                            # FINISHED TODO: Change Method to CNN

    # method_obj.to(device)

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/CNN_'  # FINISHED TODO: Change result destination folder path
    result_obj.result_destination_file_name = 'prediction_result'  # FINISHED TODO: Change result destination file name

    setting_obj = SettingGCN('Train Test Split', '')  # FINISHED TODO: Change Setting MLP to CNN

    evaluate_obj = EvaluateAccuracy('accuracy', '')  # FINISHED TODO: Fix Evaluate Accuracy to fit with CNN
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)  # FINISHED TODO: check inside base functions
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()  # FINISHED TODO: Double check dataset architecture

    epoch = [i for i in range(200)]
    print(epoch)
    print(method_obj.loss_set)
    plt.plot(epoch, method_obj.loss_set, color='green', linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor='blue', markersize=12)

    plt.title("Loss Term Values vs Training Epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss Term Values")
    plt.show()