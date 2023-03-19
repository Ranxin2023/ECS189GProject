from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Dataset_Loader import DatasetLoader
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Method_RNN import MethodRNN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Result_Saver import ResultSaver
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Setting_Train_Test_Split import SettingRNN
from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.stage_4_code.text_classification.Evaluate_Accuracy import EvaluateAccuracy
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
    data_obj = DatasetLoader('stage4_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data_medium/'
    data_obj.dataset_classification_train_pos = 'text_classification/train/pos/'
    data_obj.dataset_classification_train_neg = 'text_classification/train/neg/'
    data_obj.dataset_classification_test_pos = 'text_classification/test/pos/'
    data_obj.dataset_classification_test_neg = 'text_classification/test/neg/'

    method_obj = MethodRNN('recurrent neural network', '')

    result_obj = ResultSaver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = SettingRNN('Train Test Split IDK what is this', '')

    evaluate_obj = EvaluateAccuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)         # check inside base functions
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print(evaluate_obj.classification_report())
    print('************ Finish ************')
    # ------------------------------------------------------
    

    epoch = [i for i in range(100) if i % 10 == 0]
    print (epoch)
    plt.plot(epoch, method_obj.loss_set, color='green', linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor='blue', markersize=12)

    plt.title("Loss Term Values vs Training Epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss Term Values")

    plt.show()