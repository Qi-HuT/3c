# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.utils.data as data
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import collections
from pathlib import Path
from process_data import remodel_sentence
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import nltk
import re
import jsonlines
import matplotlib.pyplot as plt


# text = "Sentiment analysis is a challenging subject in machine learning.\
#  People express their emotions in language that is often obscured by sarcasm,\
#   ambiguity, and plays on words, all of which could be very misleading for \
#   both humans and computers. There's another Kaggle competition for movie review \
#   sentiment analysis. In this tutorial we explore how Word2Vec can be applied to \
#   a similar problem.".lower()
# text_list = nltk.word_tokenize(text)
# print(text_list)
'''
    process file for better understanding
'''

'''
    add header for arc-paper-ids
'''
# file_path = '~/Downloads/data/3C/arc-paper-ids.tsv'
# dataformat = pd.read_csv(file_path, sep='\t', names=['AnthologyID', 'Year', 'Title', 'Author'])
# print(dataformat)
# title = dataformat[dataformat['AnthologyID'] == 'A00-1004']
# print(title)
# for index, content in title.iterrows():
#     print(content['Author'])
# # data = np.array(dataformat.loc[:,:])
# # column_headers = list(dataformat.columns.values)
# print(dataformat.columns)
# # print(column_headers)
# out_path = '~/Downloads/data/3C/arc-paper-ids-have-header.tsv'
# if os.path.exists(out_path) and os.path.isfile(out_path):
#     pass
# else:
#     dataformat.to_csv('~/Downloads/data/3C/arc-paper-ids-have-header.tsv', sep='\t')

'''

'''
# annotated_json_file_path = "/home/g19tka13/Downloads/data/3C/annotated-json-data-v2/A00-1004.json"
# i = 0
# example = None
# with open(annotated_json_file_path, 'r') as f:
#     for line in f:
#         example = json.loads(line) # dict_keys(['year', 'sections', 'paper_id', 'citation_contexts'])
#
# print(type(example['year']))
# print(example['year'])
# print(type(example['sections']))
# print(len(example['sections']))  # 8
# for element in example['sections']:
#     print(element)
# print(type(example['paper_id']))
# print(type(example['citation_contexts']))

'''

'''
# python 不能将 ~ 识别为 /home/user
# json_file_path = '/home/g19tka13/Downloads/data/3C/acl-arc-json/json/A/A00/A00-1000.json'
# i = 0
# raw = None
# with open(json_file_path, 'r') as f:
#     for line in f:
#         raw = json.loads(line)
#         i = i + 1
# print(raw.keys())
data = Path('/home/g19tka13/Downloads/data/3C')
data_path = data / 'taskA/test.csv'
# da = pd.read_csv('/home/g19tka13/Downloads/data/3C/taskA/test.csv', sep=',')
da = pd.read_csv(data_path).merge(pd.read_csv(str(data_path).replace('test', 'sample_submission')), on='unique_id')
dat = pd.read_csv(str(data_path).replace('test', 'train'))
print(dat.shape[0])
count = collections.Counter(dat['citation_class_label'])
print(count)
for index, row in dat.iterrows():
    pass
str = 'They usually generate user models that describe user interests according to a set of features #AUTHOR_TAG'
repla = '12312312312312'
str = re.sub(r"#AUTHOR_TAG", repla, str)
print(str)
# print(da['citation_context'])
# dcon = pd.concat([da['citation_context'], dat['citation_context']], ignore_index=True)
# dte = da['citation_context'].append(dat['citation_context'],ignore_index=True)
# dte.rename(index={'citation_context': dte}, inplace=True)
# print(da)
# test_sen = da.loc[da['citation_class_label'] == 3]['citation_context'].reset_index(drop=True)
# sen = dat.loc[dat['citation_class_label'] == 3]['citation_context'].reset_index(drop=True)
# # print(sen)
# for i in range(len(sen)):
#     # print(i)
#     # print(sen[i])
#     print(remodel_sentence(sen[i]).split())
#     print(nltk.word_tokenize(sen[i].lower()))
# # for i in range(len(test_sen)):
#     # print(i)
#     # print(sen[i])
#     print(remodel_sentence(test_sen[i])[-5:])
# print(da.shape)
# label = []
# labelt = da['citation_class_label']
# for i in range(len(da['citation_class_label'])):
#     label.append(labelt[i])
# print(collections.Counter(label))
# input = torch.randn(3, 5)
# print(input)
# input2 = torch.randn(3, 5)
# values, indices = torch.max(input, 1)
# print(torch.max(input, 1))citation_context
# print(torch.max(input2, 1))
# target = torch.empty(3, dtype=torch.long).random_(5)

# print(torch.mean((torch.tensor(torch.max(input, -1).indices == torch.max(input2, -1).indices,dtype=torch.float))))

# out = torch.randn(10, 5)
# label = torch.randn(10)
# print(nn.CrossEntropyLoss(out, label))

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input.shape)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target.shape)
# output = loss(input, target)
# print(output)
# print(np.array([1, 2, 3]).mean())

# a = torch.randn([5, 6, 10])
# idx = torch.tensor([2, 3, 4, 1, 0])
# # print(a)
# # # print(a[:, -1, :])
# # data = a.index_select(0, idx)
# # print(data)
# out = None
# for i in range(5):
#     if out is None:
#         # print(a[i])
#         # print(a[i][0, :].shape)
#         out = torch.unsqueeze(a[i][0, :], dim=0)
#         # print(out.shape)
#     else:
#         # print(a[i][0, :])
#         out = torch.cat((out, torch.unsqueeze(a[i][0, :], dim=0)), dim=0)
#         # print(out)
#         # print(a[i][0, :])
# print(out)
#
# d2 = torch.randn([3, 4, 5])
# d3 = torch.randn([1, 4])
# d1 = torch.randn(4)
# d1 = torch.unsqueeze(d1, 1)
# # print(torch.cat([d2, d3], dim=0))
# print(d2.size())

# a = [1, 2, 3]
# b = [1, 5, 6]
# a.extend(b)
# print('a={}'.format(a))
bz=torch.Tensor([3,  1])
bz = (bz - 1).view(bz.shape[0], 1, -1)
print(bz)
bz = bz.repeat(1, 1, 2)
print(bz)
# out = torch.gather(out, 1, bz)
print(300*0.8)
a = torch.Tensor([3., 1., 0., 5., 0., 1., 4., 4., 5., 1.]).long()
b = torch.Tensor([1, 0, 1, 0, 5, 5, 5, 4, 0, 0])
out = f1_score(a, b, average='macro')
print(out)

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
x1 = torch.tensor([[0.3, 0.5, 0.7], [0.3, 0.5, 0.7]])
x2 = torch.tensor([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5]])
target = torch.tensor([[1, -1]], dtype=torch.float)
loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='none')

loss = loss_f(x1, x2, target)

print("Cosine Embedding Loss", loss)


label = torch.empty(3, dtype=torch.long).random_(5)
print(label)

emb = nn.Embedding(10 ,100)
print(emb.weight.requires_grad)

jsonpath = "/home/g19tka13/Downloads/data/3C/scicite/train.jsonl"
with jsonlines.open(jsonpath) as reader:
    for row in reader:
        if row['citingPaperId'] == '53698b91709112e5bb71eeeae94607db2aefc57c':
            print(row)
            print('true')

# x = [[1, 2],
#      [1, 1.5],
#      [2, 3],
#      [2, 4],
#      [2, 5]]
# x = np.array(x)
# center = np.array([
#           [2, 3],
#           [1, 1]])
# y_pred = KMeans(n_clusters=2, init=center).fit_predict(x)
# print(y_pred)
# plt.scatter(x[:, 0], x[:, 1], c=y_pred)
# plt.show()
matrix = np.ones(10)
print(np.unique(matrix))
b = torch.tensor(1)
print(b)
y = torch.ones(1).long()
print(y)
loss = nn.CosineEmbeddingLoss()
a1 = torch.tensor(np.array([3, 4, 3, 0, 3, 1, 5, 2, 3, 0]))
a1 = torch.unsqueeze(a1, dim=0)
print(a1)
a2 = torch.tensor(np.array([3, 4, 3, 0, 3, 1, 5, 3, 0]))
if 5 in a2:
    print('true')
# print(loss(a1, a2, y))
print(collections.Counter(np.array([3, 4, 3, 0, 3, 1, 5, 2, 3, 0]))[3])
out, count = torch.unique(a2, return_counts=True)
print(out)
print(out[2])
print(count)
print(np.log(1.02))
print(np.log(2.02))