import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

if 1:
    f = open('C:\\Users\\Admin\\Desktop\\ECS_189G\\Stage_1\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_3_data\\MNIST','rb')


   # Change MNIST to see other dataset
    data = pickle.load(f)

    f.close()

    print('training set size:', len(data['train']),
          'testing set size:', len(data['test']))

    # for pair in data['train']:
    #     plt.imshow(pair['image'], cmap="Greys")
    #     print(pair['image'].shape)
    #     plt.show()
    #     print(pair['label'])
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for pair in data['train']:
        x_train.append(np.array(pair['image']))
        y_train.append(pair['label'])
    for pair in data['test']:
        x_test.append([pair['image']])
        y_test.append([pair['label']])

    x_train = list(x_train)
    # x_train = x_train[0:59968]
    # x_train = np.split(x_train, 64)
    # print(len(x_train))
    def batch_size_generator(batch_size, lst):
        return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]
    x_train = batch_size_generator(64, x_train)

    print(x_train[0:2])
    # data['test'].reshape(1, 64, 28, 28)


