from io import open
import re
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# nltk.download('punkt')
from torch.nn.utils.rnn import pad_sequence

file1 = open('C:\\Users\\Admin\\Desktop\\ECS_189G\\Stage_1\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_4_data\\text_generation\data', 'r')
Lines = file1.readlines()
# #

def data_cleaner(data, index):
    if index == 41 or index == 679 or index == 750:
        return re.sub(r'\($' ,'', data)
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
tokenizer_data = []
words_dictonary = {}


def get_dict(data):
    for sentence in tokenizer_data:
        for word in sentence:
            if word in words_dictonary:
                words_dictonary[word] += 1
            else:
                words_dictonary[word] = 1
    sorted_dict = sorted(words_dictonary.items(), key=lambda x:x[1], reverse=True)
    unique_word = [pair[0] for pair in sorted_dict]
    dictionary = {word: index for index, word in enumerate(unique_word)}
    return dictionary

data = []

for i in range(1, len(Lines)):
    # print(line.strip().split())
    line = Lines[i].replace('"', '')
    line = line.split(',', 1)[1]

    # line  = line.strip().split()
    # punctuation= "!.?"
    line = re.sub(r'http\S+', '', line)
    line = data_cleaner(line, i)

    data.append(line)


for i in range(len(data)):
    tokens = re.findall(r'\w+|[^\w\s]', data[i])
    tokenizer_data.append(tokens)

dictonary = get_dict(tokenizer_data)
def words_to_index(data):
    word_index = []
    for i in range(5):
        index = []
        for j in range(len(data[i])):
            index.append(dictonary[data[i][j]])

        word_index.append(index)
    return word_index


print(words_to_index(tokenizer_data))




