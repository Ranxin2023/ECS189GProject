"""
Concrete IO class for a specific dataset
"""
import torch

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from Stage_1.ECS189G_Winter_2022_Source_Code_Template.code.base_class.dataset import BaseDataset
import pandas as pd
import numpy as np
import string
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DatasetLoader(BaseDataset):
    data = None
    dataset_source_folder_path = None
    dataset_classification_train_pos = None
    dataset_classification_train_neg = None
    dataset_classification_test_pos = None
    dataset_classification_test_neg = None
    padding = 100
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        word_vectors = self.getWordVectors(self.dataset_source_folder_path)
        # List all directory paths
        train_pos_path = self.dataset_source_folder_path + self.dataset_classification_train_pos
        train_neg_path = self.dataset_source_folder_path + self.dataset_classification_train_neg
        test_pos_path = self.dataset_source_folder_path + self.dataset_classification_test_pos
        test_neg_path = self.dataset_source_folder_path + self.dataset_classification_test_neg

        # In each directory, the list of files it contained
        train_file_pos = os.listdir(train_pos_path)
        train_file_neg = os.listdir(train_neg_path)
        test_file_pos = os.listdir(test_pos_path)
        test_file_neg = os.listdir(test_neg_path)

        # Clean the word
        train_pos_clean = self.cleanWords(train_file_pos, train_pos_path, word_vectors)
        train_neg_clean = self.cleanWords(train_file_neg, train_neg_path, word_vectors)
        test_pos_clean = self.cleanWords(test_file_pos, test_pos_path, word_vectors)
        test_neg_clean = self.cleanWords(test_file_neg, test_neg_path, word_vectors)

        # Word embedding
        train_pos_vector = self.wordEmbed(train_pos_clean, word_vectors)
        train_neg_vector = self.wordEmbed(train_neg_clean, word_vectors)
        test_pos_vector = self.wordEmbed(test_pos_clean, word_vectors)
        test_neg_vector = self.wordEmbed(test_neg_clean, word_vectors)

        # Put the data into x_train and y_train
        x_train = []
        x_train.extend(train_pos_vector)
        x_train.extend(train_neg_vector)
        x_train = np.array(x_train)
        # 1 denote as positive, 0 denote as negative
        y_train = []
        y_pos = [1] * len(train_pos_vector)
        y_neg = [0] * len(train_neg_vector)
        y_train.extend(y_pos)
        y_train.extend(y_neg)
        y_train = np.array(y_train)

        # Put the data into x_test and y_test
        x_test = []
        x_test.extend(test_pos_vector)
        x_test.extend(test_neg_vector)
        x_test = np.array(x_test)
        # 1 denote as positive, 0 denote as negative
        y_pos = [1] * len(test_pos_vector)
        y_neg = [0] * len(test_neg_vector)
        y_test = []
        y_test.extend(y_pos)
        y_test.extend(y_neg)
        y_test = np.array(y_test)

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    def cleanWords(self, file_dict, file_direct, word_vectors):
        clean_word_list = []
        print(file_dict)
        for filename in file_dict:
            # read the file and extract the data
            file_path = file_direct + filename
            file = open(file_path, 'rt')
            text = file.read()
            file.close()
            # split into words
            word_tokens = word_tokenize(text)
            # convert to lower case
            word_tokens = [word.lower() for word in word_tokens]
            # delete punctuations by string.punctuation
            no_punc_tokens = [word for word in word_tokens if word not in string.punctuation]
            # delete words that do not contain alphabets
            eng_words = [word for word in no_punc_tokens if word.isalpha()]
            # delete stop words
            stop_words = set(stopwords.words('english'))
            nonstop_words = [word for word in eng_words if not word in stop_words]
            clean_words = [word for word in nonstop_words if word in word_vectors]
            # change all words into their original form (stem form)
            clean_word_list.append(clean_words)
        return clean_word_list

    def getWordVectors(self, dataset_source_folder_path):
        output = {}
        file = open(dataset_source_folder_path+"text_classification/glove.6B.200d.txt", 'r')
        for line in file:
            tokens = line.split()
            word = tokens[0]
            word_vector = np.array(tokens[1:])
            vector_normalized = word_vector.astype('float32')
            output[word] = vector_normalized
        return output

    def wordEmbed(self, word_array, word_vectors):
        output = []
        for file in word_array:
            file_output = []
            for word in file:
                file_output.append(word_vectors[word])
            file_output = self.padding_resize(file_output, self.padding)
            output.append(file_output)
        output_vector = np.array(output)
        return output_vector

    def padding_resize(self, word_array, size):
        output = []
        if len(word_array) >= size:
            output = word_array[0:size]
        else:
            output.extend(word_array)
            loop_time = size - len(word_array)
            zero_arr = np.array([0] * 200)
            zero_arr = zero_arr.astype('float32')
            for i in range(loop_time):
                output.append(zero_arr)
        output_arr = np.array(output)
        return output_arr

    def reshape_vector(self, vectors):
        result = []
        for vector in vectors:
            vector = vector.reshape(2, 10, 5800)
            result.append(vector)
        result = np.array(result)
        return result
