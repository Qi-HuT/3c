# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os.path
from pathlib import Path
import nltk
# import findsenten
from torchtext.vocab import Vocab, Vectors
import re
import collections
import torch.nn as nn
import torch.utils.data as Data
import torch


# def remodel_sentence(sentence):
#     # sentence = re.sub(r"#Author_Tag", "#Author#Tag", sentence)
#     sentence = findsenten.sub(r"[^A-Za-z0-9(),:!?_\'\`#]", " ", sentence)
#     sentence = findsenten.sub(r":", ' : ', sentence)
#     sentence = findsenten.sub(r"\'s", " \'s", sentence)
#     sentence = findsenten.sub(r"\'ve", " \'ve", sentence)
#     sentence = findsenten.sub(r"n\'t", " n\'t", sentence)
#     sentence = findsenten.sub(r"\'re", " \'re", sentence)
#     sentence = findsenten.sub(r"\'d", " \'d", sentence)
#     sentence = findsenten.sub(r"\'ll", " \'ll", sentence)
#     sentence = findsenten.sub(r",", " , ", sentence)
#     sentence = findsenten.sub(r"!", " ! ", sentence)
#     sentence = findsenten.sub(r"\(", " ( ", sentence)
#     sentence = findsenten.sub(r"\)", " ) ", sentence)
#     sentence = findsenten.sub(r"\?", " ? ", sentence)
#     sentence = findsenten.sub(r"\s{2,}", " ", sentence)
#     return sentence.lower()


def strtolist():
    '''
            load train val data
    :return: the content of data str to list
    '''
    print('Loading data')
    data_path = Path('/home/g19tka13/taskA')
    taskA_data = data_path / 'train.csv'
    # taskA_data = None
    df = pd.read_csv(taskA_data, sep=',')
    df = df.sample(frac=1).reset_index(drop=True)
    total_instance_num = float(df.shape[0])
    print(df.shape)
    # df = df.head()
    # print(df['citation_context'])
    # for context in df['citation_context']: # 只是遍历其中一列的值
    #     print(context)
    df_header = df.columns
    train_data = pd.DataFrame(columns=list(df_header))
    # class_array = np.zeros(6, dtype=np.int64)
    class_num = collections.Counter(df['citation_class_label'])  # Counter.items
    weighted = np.zeros(6, dtype=np.float32)
    # 保留小数位数 使用%.2f，round函数或者float函数。假设要保留两位小数
    # round(num, 2) || '%.2f' % num || float('%.2f' % num)
    for i in range(6):
        weighted[i] = np.log10((total_instance_num - class_num[i]) / class_num[i]) + 1
    for index, raw in df.iterrows():
        # print(getattr(raw, 'citation_context').split())
        # print(remodel_sentence(getattr(raw, 'citation_context')).split())
        # print(raw['citation_context'])
        # print(remodel_sentence(raw['citation_context']).split())
        # df.loc[index, 'citation_context'] = remodel_sentence(raw['citation_context']).split()  # 修改值错误,因为两者长度不一致.
        # class_array[raw['citation_class_label']] = 1
        # if raw['citation_class_label'] not in
        train_data.loc[index] = {"unique_id": raw['unique_id'], 'core_id': raw['core_id'], 'citing_title': raw['citing_title'],
                                 'citing_author': raw['citing_author'], 'cited_title': raw['cited_title'], 'cited_author': raw['cited_author'],
                                 'citation_context': nltk.word_tokenize(raw['citation_context'].lower()), 'citation_class_label': raw['citation_class_label']}
        # class_array[raw['citation_class_label']] = 0
        # print(remodel_sentence(raw['citation_context']))
        # print(nltk.word_tokenize(getattr(raw, 'citation_context')))
    return train_data, weighted


def loadtestdata():
    print('Loading test data')
    data_path = Path('/home/g19tka13/taskA')
    taskA_test_data = data_path / 'test.csv'
    test_df = pd.read_csv(taskA_test_data, sep=',').merge(pd.read_csv(str(taskA_test_data).replace('test', 'sample_submission')), on='unique_id')
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    print(test_df.shape)
    # test_df = test_df.loc[test_df['citation_class_label'] == 0].reset_index(drop=True)  # drop去除原来的索引
    test_df_header = test_df.columns
    test_data = pd.DataFrame(columns=list(test_df_header))
    label = ['background', 'compares', 'contrasts', 'extension', 'future', 'motivation', 'uses']
    for index, raw in test_df.iterrows():
        label_word = str(label[raw['citation_class_label']])
        citation_text = re.sub(r"#AUTHOR_TAG", label_word, raw['citation_context'].lower())
        citation_text = nltk.word_tokenize(citation_text)
        # print(label_word)
        test_data.loc[index] = {"unique_id": raw['unique_id'], 'core_id': raw['core_id'], 'citing_title': raw['citing_title'],
                                'citing_author': raw['citing_author'], 'cited_title': raw['cited_title'], 'cited_author': raw['cited_author'],
                                # 'citation_context': nltk.word_tokenize(raw['citation_context'].lower()), 'citation_class_label': raw['citation_class_label']}
                                'citation_context': citation_text,
                                'citation_class_label': raw['citation_class_label']}

        # remodel_sentence(raw['citation_context']).split()
    return test_data




# def assemble(data, vocabulary, num):
#
#     '''
#         将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
#         同时将sentence和label和sentence_len组装起来方便载入。
#     '''
#     if num == 1:
#         text_data = data['citation_context']
#         label_data = data['citation_class_label']
#         print('train{}'.format(collections.Counter(label_data)))
#         num_text = len(text_data)
#         text_len = np.array([len(sentence) for sentence in text_data])
#         max_text_len = max(text_len)
#         train_sen_len = []
#         val_sen_len = []
#         train_label = []
#         val_label = []
#         # wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(text_data), max_text_len], dtype=np.int64)  # word_to_index
#         train_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.8), max_text_len], dtype=np.int64)  # word_to_index
#         val_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.2), max_text_len], dtype=np.int64)
#         for i in range(len(text_data)):
#             if i <= int(num_text * 0.8) - 1:
#                 train_sen_len.append(len(text_data[i]))
#                 train_label.append(label_data[i])
#                 train_wtoi_matrix[i, :len(text_data[i])] = [
#                     vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
#                     for word in text_data[i]]
#             else:
#                 val_sen_len.append(len(text_data[i]))
#                 val_label.append(label_data[i])
#                 val_wtoi_matrix[i - int(num_text * 0.8), :len(text_data[i])] = [
#                     vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
#                     for word in text_data[i]]
#             # j += 1
#             # if j > 49:
#             #     break
#
#         # return wtoi_matrix
#         train_iter = Data.TensorDataset(torch.from_numpy(train_wtoi_matrix), torch.Tensor(train_sen_len), torch.Tensor(train_label))
#         val_iter = Data.TensorDataset(torch.from_numpy(val_wtoi_matrix), torch.Tensor(val_sen_len), torch.Tensor(val_label))
#         # return (train_iter, val_iter) if num == 1 else False
#         return train_iter, val_iter
#     else:
#         text_data = data['citation_context']
#         label_data = data['citation_class_label']
#         print('test{}'.format(collections.Counter(label_data)))
#         text_len = np.array([len(sentence) for sentence in text_data])
#         max_text_len = max(text_len)
#         test_sen_len = []
#         test_label = []
#         test_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(text_data), max_text_len], dtype=np.int64)
#         for i in range(len(text_data)):
#             test_sen_len.append(len(text_data[i]))
#             test_label.append(label_data[i])
#             test_wtoi_matrix[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
#                                                        for word in text_data[i]]
#         test_iter = Data.TensorDataset(torch.from_numpy(test_wtoi_matrix), torch.Tensor(test_sen_len), torch.Tensor(test_label))
#         return test_iter


def assemble(data, vocabulary, num, prelabeled_data=None, unlabeled_data=None):

    '''
        将每个句子中的每个单词映射成词汇表中单词对应的id  word->id
        同时将sentence和label和sentence_len组装起来方便载入。
    '''
    unlabeled_text =None
    preunlabel_data = None
    unlabel_iter = None
    max_text_len_final = None
    label = ['background', 'compares', 'contrasts', 'extension', 'future', 'motivation', 'uses']
    if num == 1:
        text_data = data['citation_context']
        label_data = data['citation_class_label']
        print('train{}'.format(collections.Counter(label_data)))
        num_text = len(text_data)
        text_len = np.array([len(sentence) for sentence in text_data])
        max_text_len = max(text_len)
        train_sen_len = []
        val_sen_len = []
        train_label = []
        val_label = []
        # wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(text_data), max_text_len], dtype=np.int64)  # word_to_index
        if prelabeled_data is not None:
            # max_text_len = if np.array([len(sentence) for sentence in text_data])
            unlabeled_text = prelabeled_data['citation']  # 外部数据引文
            preunlabel_data = prelabeled_data['citation_class_label'] # 外部数据标签
            unlabeled_text_max = max(np.array([len(sentence) for sentence in unlabeled_text]))  # 外部引文的最大长度
            max_text_len_final = max_text_len if unlabeled_text_max < max_text_len else unlabeled_text_max # 比较训练集和外部引文的最大长度
            train_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones(
                [int(num_text * 0.8) + len(unlabeled_text), max_text_len_final], dtype=np.int64)  # word_to_index
            label_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.8) + len(unlabeled_text), 1], dtype=np.int64)
        else:
            train_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.8), max_text_len], dtype=np.int64)  # word_to_index
            label_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.8), 1], dtype=np.int64)
        val_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([int(num_text * 0.2), max_text_len], dtype=np.int64)
        for i in range(len(text_data)):
            if i <= int(num_text * 0.8) - 1:
                train_sen_len.append(len(text_data[i]))
                train_label.append(label_data[i])
                label_wtoi_matrix[i, :1] = [vocabulary.stoi[label[label_data[i]]] if label[label_data[i]] in vocabulary.stoi else vocabulary.stoi['<unk>']]
                                            # for word in label[label_data[i]]]
                train_wtoi_matrix[i, :len(text_data[i])] = [
                    vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                    for word in text_data[i]]
            else:
                val_sen_len.append(len(text_data[i]))
                val_label.append(label_data[i])
                val_wtoi_matrix[i - int(num_text * 0.8), :len(text_data[i])] = [
                    vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                    for word in text_data[i]]
        if prelabeled_data is not None:
            length = int(num_text * 0.8)
            print(train_wtoi_matrix.shape)
            for i in range(int(num_text * 0.8), train_wtoi_matrix.shape[0]):
                train_sen_len.append(len(unlabeled_text[i - length]))
                train_label.append(preunlabel_data[i - length])
                label_wtoi_matrix[i, :1] = [
                    vocabulary.stoi[label[preunlabel_data[i - length]]] if label[preunlabel_data[i - length]] in vocabulary.stoi else vocabulary.stoi['<unk>']]
                train_wtoi_matrix[i, :len(unlabeled_text[i - length])] = [vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>'] for word in unlabeled_text[i - length]]
        print(torch.from_numpy(train_wtoi_matrix).shape, torch.Tensor(train_sen_len).shape, torch.Tensor(train_label).shape, torch.from_numpy(label_wtoi_matrix).shape)
        train_iter = Data.TensorDataset(torch.from_numpy(train_wtoi_matrix), torch.Tensor(train_sen_len), torch.Tensor(train_label), torch.from_numpy(label_wtoi_matrix))
        val_iter = Data.TensorDataset(torch.from_numpy(val_wtoi_matrix), torch.Tensor(val_sen_len), torch.Tensor(val_label))
        return train_iter, val_iter, np.unique(label_wtoi_matrix)
    else:
        text_data = data['citation_context']
        label_data = data['citation_class_label']
        print('test{}'.format(collections.Counter(label_data)))
        text_len = np.array([len(sentence) for sentence in text_data])
        max_text_len = max(text_len)
        test_sen_len = []
        test_label = []
        test_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(text_data), max_text_len], dtype=np.int64)
        for i in range(len(text_data)):
            test_sen_len.append(len(text_data[i]))
            test_label.append(label_data[i])
            test_wtoi_matrix[i, :len(text_data[i])] = [vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                                                       for word in text_data[i]]
        if unlabeled_data is not None:
            unlabeled_sen_len = []
            unlabeled_data_index = []
            unlabeled_text = unlabeled_data['citation']
            max_unlabeled_text = max(np.array([len(sentence) for sentence in unlabeled_text]))
            unlabeled_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(unlabeled_text), max_unlabeled_text], dtype=np.int64)
            for index, row in unlabeled_data.iterrows():
                unlabeled_sen_len.append(len(row['citation']))
                unlabeled_data_index.append(index)
                unlabeled_wtoi_matrix[index, :len(row['citation'])] = [
                    vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
                    for word in row['citation']]
            unlabel_iter = Data.TensorDataset(torch.from_numpy(unlabeled_wtoi_matrix), torch.Tensor(unlabeled_sen_len),
                                              torch.Tensor(unlabeled_data_index))
            print(unlabel_iter)
        print('shape', test_wtoi_matrix.shape)
        test_iter = Data.TensorDataset(torch.from_numpy(test_wtoi_matrix), torch.Tensor(test_sen_len), torch.Tensor(test_label))
        if unlabel_iter is None:
            return test_iter, 'None'
        else:
            return test_iter, unlabel_iter
        # return (test_iter, None) if unlabel_iter is None else (test_iter, unlabel_iter)

# def labeltoid(label, vocabulary):
#     label_wtoi_matrix = vocabulary.stoi['<pad>'] * np.ones([len(label), 1], dtype=np.int64)
#     for i in range(len(label)):
#         label_wtoi_matrix[i, :] = [vocabulary.stoi[word] if word in vocabulary.stoi else vocabulary.stoi['<unk>']
#                                    for word in label]
#     return label_wtoi_matrix


def count_all_words(data):
    words = []
    for row in data:
        words.extend(row)
    return words


def load_word_vector(train_data, test_data, train_type=None, used_unlabeled_data=None):

    label_vector = pd.Series(['background', 'compares', 'contrasts', 'extension', 'future', 'motivation', 'uses'])
    # Download word vector
    print('Loading word vectors')
    path = os.path.join('/home/g19tka13/wordvector', 'wiki.en.vec')
    if not os.path.exists(path):
        print('Download word vectors')
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
                                   path)
    vectors = Vectors('wiki.en.vec', cache='/home/g19tka13/wordvector')
    if train_type == 'self_train':
    # print(train_data['citation_context'].append(test_data['citation_context'], ignore_index=True))
        print(used_unlabeled_data['citation'])
        vocab = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append(
            test_data['citation_context'], ignore_index=True).append(label_vector, ignore_index=True).append(
            used_unlabeled_data['citation'], ignore_index=True))), specials=['<pad>', '<unk>'], vectors=vectors)
        return vocab
    else:
        vocab = Vocab(collections.Counter(count_all_words(train_data['citation_context'].append(
            test_data['citation_context'], ignore_index=True).append(label_vector, ignore_index=True))),
                      specials=['<pad>', '<unk>'], vectors=vectors)
        return vocab


# def load_data(train_type=None, id_list=None, prelabeled_data=None, unlabeled_data=None):
#
#     train_data, weighted = strtolist()
#     test_data = loadtestdata()
#     if train_type == 'self_train':
#         vocab = load_word_vector(train_data, test_data, 'self_train', unlabeled_data)
#
#         if id_list is None:  # 判断是否第一次训练模型。
#             train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
#         else:
#             train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1, prelabeled_data)  # 加入数据
#     else:
#         vocab = load_word_vector(train_data, test_data)
#         train_iter, val_iter, label_word_id = assemble(train_data, vocab, 1)
#     test_iter = assemble(test_data, vocab, 0)
#     return train_iter, val_iter, test_iter, vocab, weighted, label_word_id


def split_unlabeled_data(unlabeled_data, id_list, class_id, old_data=None):
    print(unlabeled_data)
    print(len(id_list), id_list)
    prelabel_data = unlabeled_data[unlabeled_data.index.isin(id_list)]
    print(prelabel_data)
    print(len(class_id), class_id)
    prelabel_data.insert(prelabel_data.shape[1], 'citation_class_label', class_id)
    print(prelabel_data)
    # prelabel_data.loc[:, 'citation_class_label'] = class_id
    if old_data is None:
        prelabel_data = prelabel_data.reset_index(drop=True)
    else:
        prelabel_data = pd.concat([old_data, prelabel_data], axis=0).reset_index(drop=True)
    unlabeled_data = unlabeled_data[~unlabeled_data.index.isin(id_list)].reset_index(drop=True)
    print(prelabel_data)
    return unlabeled_data, prelabel_data




