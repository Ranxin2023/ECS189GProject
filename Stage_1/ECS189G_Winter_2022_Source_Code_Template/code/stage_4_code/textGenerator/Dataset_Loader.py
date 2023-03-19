"""
Concrete IO class for a specific dataset
"""
import numpy as np
import pickle
from sklearn.utils import shuffle
from io import open
from keras.preprocessing.text import Tokenizer
import re
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.dataset import BaseDataset
import pandas as pd

class DatasetLoader(BaseDataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    words_dictionary = {}
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def create_dictionary(self, dictionary):
        self.words_dictionary = dictionary

    def load(self):
        print('loading data...')
        dictionary = {}
        tokenizer_data = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        Lines = f.readlines()
        data = []
        for i in range(1, len(Lines)):
            # print(line.strip().split())
            line = Lines[i].replace('"', '')
            line = line.split(',', 1)[1]

            # line  = line.strip().split()
            # punctuation= "!.?"
            line = re.sub(r'http\S+', '', line)
            line = self.data_cleaner(line, i)
            line = line.lower()
            line = re.sub(r'[^\w\s]','', line)
            data.append(line)
        self.prepare_tokenizer_data(data, tokenizer_data)
        word_dictionary = self.get_dict(tokenizer_data, dictionary)
        self.create_dictionary(word_dictionary)
        x_train = self.words_to_index(tokenizer_data, word_dictionary)
        # x_train = self.batch_size(x_train[0:1600])
        x_train = self.batch_size(x_train[0:23000])

        return {'x_train': x_train, 'dictionary': word_dictionary}

    def prepare_tokenizer_data(self, data, tokenizer_data):
        for i in range(len(data)):
            tokens = re.findall(r'\w+|[^\w\s]', data[i])
            tokenizer_data.append(tokens)

    def words_to_index(self, data, word_dictionary):
        word_index = []
        for i in range(len(data)):

            for j in range(len(data[i])):
                word_index.append(word_dictionary[data[i][j]])


        return word_index

    def get_dict(self, tokenizer_data, dictionary):
        for sentence in tokenizer_data:
            for word in sentence:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        unique_word = [pair[0] for pair in sorted_dict]
        word_dictionary = {word: index for index, word in enumerate(unique_word)}
        return word_dictionary

    def data_cleaner(self, data, index):
            if index == 41 or index == 679 or index == 750:
                return re.sub(r'\($', '', data)
            elif index == 212:
                return re.sub(r'\(X-post from /r/Jokes\)', '', data)
            elif index == 264:
                return re.sub(r'\(X-post from /r/jokes\)', '', data)

            elif index == 232:
                return re.sub(r'From.*', '', data)
            elif index == 247:
                return re.sub('^X-post[^:]+: ', '', data)
            elif index == 870:
                return re.sub(r'\[X post from /r/Fantasy\]', '', data)

            elif index == 1322:
                line = re.sub(r'\(x-post from /r/3amjokes\)', '', data)
                line = re.sub(r'\(.*', '', line)
                return line
            elif index == 1398:
                return re.sub(r'\[.*', '', data)
            elif index == 1608:
                return re.sub(r'\(x-post from /r/jokes\)', '', data)
            elif index == 1619:
                return re.sub(r'\[x-post from r/Jokes\]', '', data)
            else:
                return data

    def batch_size(self, data):
        return np.array(data).reshape(100,46,5)