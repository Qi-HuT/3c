# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os.path
from pathlib import Path
import nltk
import re
from torchtext.vocab import Vocab, Vectors
import collections
import torch.nn as nn
import torch.utils.data as Data
import torch


def remodel_sentence(sentence):
    # sentence = re.sub(r"#Author_Tag", "#Author#Tag", sentence)
    sentence = re.sub(r"[^A-Za-z0-9(),:!?_\'\`#]", " ", sentence)
    sentence = re.sub(r":", ' : ', sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.lower()


def strtolist():
    print('Loading data')
    data_path = Path('/home/g19tka13/Downloads/data/3C')
    taskA_data = data_path / 'taskA/train.csv'
    # taskA_data = None
    df = pd.read_csv(taskA_data, sep=',')
    # df = df.head()
    # print(df['citation_context'])
    # for context in df['citation_context']: # 只是遍历其中一列的值
    #     print(context)
    df_header = df.columns
    df1 = pd.DataFrame(columns=list(df_header))
    # class_array = np.zeros(6, dtype=np.int64)
    for index, raw in df.iterrows():
        # print(getattr(raw, 'citation_context').split())
        # print(remodel_sentence(getattr(raw, 'citation_context')).split())
        # print(raw['citation_context'])
        # print(remodel_sentence(raw['citation_context']).split())
        # df.loc[index, 'citation_context'] = remodel_sentence(raw['citation_context']).split()  # 修改值错误,因为两者长度不一致.
        # class_array[raw['citation_class_label']] = 1
        df1.loc[index] = {"unique_id": raw['unique_id'], 'core_id': raw['core_id'], 'citing_title': raw['citing_title'],
                          'citing_author': raw['citing_author'], 'cited_title': raw['cited_title'], 'cited_author': raw['cited_author'],
                          'citation_context': remodel_sentence(raw['citation_context']).split(), 'citation_class_label': raw['citation_class_label']}
        # class_array[raw['citation_class_label']] = 0
        # print(remodel_sentence(raw['citation_context']))
        # print(nltk.word_tokenize(getattr(raw, 'citation_context')))
    return df1


def count_all_words(data):
    words = []
    for row in data:
        words.extend(row)
    return words


def assemble(data, vocabulary):
    '''
        将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
        同时将sentence和label和sentence_len组装起来方便载入。
    '''
    j = 0
    text_data = data['citation_context']
    label_data = data['citation_class_label']
    text_len = np.array([len(sentence) for sentence in text_data])
    max_text_len = max(text_len)
    train_sen_len = []
    val_sen_len = []
    train_label = []
    val_label = []
    # wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(text_data), max_text_len], dtype=np.int64)  # word_to_index
    train_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([2900, max_text_len], dtype=np.int64)  # word_to_index
    val_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([100, max_text_len], dtype=np.int64)
    print(len((text_data)))
    for i in range(len(text_data)):
        if i <= 2899:
            train_sen_len.append(len(text_data[i]))
            train_label.append(label_data[i])
            train_wtoi_matrix[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                                                        for word in text_data[i]]
        else:
            val_sen_len.append(len(text_data[i]))
            val_label.append(label_data[i])
            val_wtoi_matrix[i - 2900, :len(text_data[i])] = [
                vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                for word in text_data[i]]
            # j += 1
            # if j > 49:
            #     break

    # return wtoi_matrix
    train_iter = Data.TensorDataset(torch.from_numpy(train_wtoi_matrix), torch.Tensor(train_sen_len), torch.Tensor(train_label))
    val_iter = Data.TensorDataset(torch.from_numpy(val_wtoi_matrix), torch.Tensor(val_sen_len), torch.Tensor(val_label))
    return train_iter, val_iter


def load_word_vector(data):
    # Download word vector
    print('Loading word vectors')
    path = os.path.join('/home/g19tka13/Downloads/data/wordvector', 'wiki.en.vec')
    if not os.path.exists(path):
        print('Download word vectors')
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
                                   path)
    vectors = Vectors('wiki.en.vec', cache='/home/g19tka13/Downloads/data/wordvector')
    vocab = Vocab(collections.Counter(count_all_words(data['citation_context'])), specials=['<pad>', '<unk>'], vectors=vectors)
    return vocab


def load_data():
    data = strtolist()
    vocab = load_word_vector(data)
    train_iter, val_iter = assemble(data, vocab)
    return train_iter, val_iter, vocab

# class LSTM(nn.module):
#     pass




