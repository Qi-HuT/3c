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
# from process_data import remodel_sentence
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
# import nltk
# import findsenten
# import jsonlines
import re
# import matplotlib.pyplot as plt


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
# data = Path('/home/g19tka13/Downloads/data/3C')
# data_path = data / 'taskB/test.csv'
# # da = pd.read_csv('/home/g19tka13/Downloads/data/3C/taskA/test.csv', sep=',')
# test = pd.read_csv(data_path, sep='\t')
# print(90*'*',test.shape)
# da = pd.read_csv(data_path).merge(pd.read_csv(str(data_path).replace('test', 'sample_submission')), on='unique_id')
# dat = pd.read_csv(str(data_path).replace('test', 'train'))
# print(dat.shape[0])
# # count = collections.Counter(dat['citation_class_label'])
# # print(count)
# for index, row in dat.iterrows():
#     pass
# str = 'They usually generate user models that describe user interests according to a set of features #AUTHOR_TAG'
# repla = '12312312312312'
# # str = findsenten.sub(r"#AUTHOR_TAG", repla, str)
# print(str)
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
# i = 0
# jsonpath = "/home/g19tka13/Downloads/data/3C/CORE_paper_list/mash/acl-arc/test.jsonl"
# section_name = {}
# with jsonlines.open(jsonpath) as reader:
#     for row in reader:
#         # print(row.keys())
#         print(row['section_title'])
#         # print(row['section_name'])
#         # if row['citingPaperId'] == '53698b91709112e5bb71eeeae94607db2aefc57c':
#         #     print(row)
#         #     print('true')
#         # i = i + 1
#         if row['section_name'] in section_name.keys():
#             section_name[row['section_name']] += 1
#         else:
#             section_name[row['section_name']] = 1
# print(section_name)
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
# matrix = np.ones(10)
# print(np.unique(matrix))
# b = torch.tensor(1)
# print(b)
# y = torch.ones(1).long()
# print(y)
# loss = nn.CosineEmbeddingLoss()
# a1 = torch.tensor(np.array([3, 4, 3, 0, 3, 1, 5, 2, 3, 0]))
# a1 = torch.unsqueeze(a1, dim=0)
# print(a1)
# a2 = torch.tensor(np.array([3, 4, 3, 0, 3, 1, 5, 3, 0]))
# if 5 in a2:
#     print('true')
# # print(loss(a1, a2, y))
# print(collections.Counter(np.array([3, 4, 3, 0, 3, 1, 5, 2, 3, 0]))[3])
# out, count = torch.unique(a2, return_counts=True)
# print(out)
# print(out[2])
# print(count)
# print(np.log(1.02))
# print(np.log(2.02))


# b = torch.zeros(5, 4, 3, 1)
# a = torch.ones((1, 10))
# print(a.softmax(dim=1))
# print(b.shape)
# c = b.softmax(dim=1)
# print(c)

a = 'CREATIVITY RESEARCH IS TYPICALLY FOCUSED'
print(re.match(r'[0-9]\.*(\s*)[A-Za-z]', a))
print(re.match(r'^([A-Z]+\s*){,4}[a-zA-Z]*', a))
# b= '   '
# print(len(b))
# print(b.split(' '))

# c = ['Methodology', '']
# print(c[1].isspace())
# section = list(filter(lambda q: re.match(r'^[A-Za-z]*$', q) and q != '', c))
# print(section)

# for i in range(4, 8):
#     print(i)
df = pd.DataFrame(np.arange(20).reshape(5, 4), columns=['a', 'b', 'c', 'd'], index=range(0, 5))
print(df)
df.loc[:, 'new'] = [1, 2, 3, 4, 5]
print(df)
a = df[df.index.isin([3, 4])]
a['tt'] = [9 ,10]
print(a)



tensor_3d = torch.LongTensor([1, 2, 3, 4, 5, 6])
tensor_3d = tensor_3d.view(6, 1, -1)
cat = tensor_3d.repeat(1, 1, 6)
print(cat)
ten_list = torch.randn(6, 7, 6)
print(ten_list)
final_result = torch.gather(ten_list, 1, cat)
print(torch.gather(ten_list, 1, cat), final_result.shape)
print(torch.squeeze(final_result, dim=1), torch.squeeze(final_result, dim=1).shape)
cut_test = torch.LongTensor([1, 2, 3, 4, 5, 6])
print(cut_test - 1)
t = []
for i in range(6):
    t.append([i])
print(t)

# print(torch.cuda.is_available())
# x_cpu = torch.ones((3, 3))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("x_cpu:\ndevice: {} is_cuda: {} id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))
#
# x_gpu = x_cpu.to(device)
# print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))
c = torch.LongTensor([1 ,2, 3, 4, 5])
print(c.tolist())
# a = None
# for i in range(5):
#     # if a is None:
#     #     a = c
#     # else:
#
#         a = torch.cat((a, c), 0)
# print(a)
print(torch.square(c))
testmodel = MLPClassifier(hidden_layer_sizes=(256, 256, 128))
print(testmodel)

m = torch.randn(5, 16, 50, 100)
cov = nn.Conv2d(16, 256, 9, stride=1)
cov2 = nn.Conv2d(256, 32, (9, 9), stride=2)
out = cov(m)
out1 = cov2(out)
print(out.shape)
print(out1.shape)
# print(torch.zeros(1, 32*6*6, 10, 1))
t1 = torch.randn(5, 8, 32, 6, 6)  # u
print(t1.shape)
t1 = t1.view(5, 32 * 6 * 6, -1)
print(t1.shape)

b = torch.zeros(1, 3, 4, 1)
print(b.softmax(dim=1))
print(b.softmax(dim=1).sum(dim=2))
print(b.softmax(dim=2))
print(b.softmax(dim=2).sum(dim=2))
a = torch.cat([b.softmax(dim=1)] * 3, dim=0).unsqueeze(4)
t2 = torch.randn(5, 2, 4, 4)
print(t2)
print(t2.contiguous().view(5, -1, 2))
print(t2.reshape(5, -1, 2))
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

print(squash(t2.reshape(5, -1, 2)))
# (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1)

a = torch.randn(1, 2, 10, 16, 8)
b = torch.randn(5, 1, 10, 8, 1)
print(torch.matmul(a, b).shape)
t = torch.randn(2, 3, 1)
print(t)
print(torch.squeeze(t))
intran = torch.rand(2, 3)
print(intran)
at = torch.square(intran).sum(1 ,keepdim=True)
print(at)
print(at.sqrt())
print(at.sqrt().max(dim=0))
print(torch.norm(intran, dim=1, keepdim=True))

id = torch.LongTensor([1., 1., 5., 5., 0., 0., 0., 4., 0., 0.])
eye_test = torch.eye(10, 6).index_select(dim=0, index=id)
print(eye_test)

tmatrix = torch.randint(1, 5, (3, 3))
print(tmatrix)
tmatrix1 = torch.randint(2, 3, (3,3))
print(tmatrix1)
print(tmatrix * tmatrix1)

a = [1 ,2, 3, 4]
for index, num in enumerate(a, 0):
    print(index)

a = torch.rand(3, 10, 1)
print(a)
print(a[0])
print(torch.arange(0, 5))
print(torch.index_select(a[0], 0, torch.arange(0, 5)))
print(a)
# ma = max((5, ))
value, index = torch.topk(a[0], k=5, dim=0, largest=True)
print(torch.topk(a[0], k=5, dim=0, largest=True))
print(torch.squeeze(index))
print(torch.index_select(a[0], 0, torch.squeeze(index)))
print('value', value)
print('123',torch.squeeze(value).unsqueeze(dim=0))
a = torch.randn(3, 4)
print(a)
print(torch.sum(a, dim=0))
a = torch.Tensor([1, 2, 3])
b = torch.Tensor([4, 5, 6])
li = []
li.append(a)
li.append(b)
print(li)
print(torch.stack(li))

one = torch.ones((10, 1), dtype=torch.float)
print(one)

idx = list(range(100))
y = list(range(100))
print('idx',idx)
num_step = 100 // 10 + 1
by = None
for i in range(num_step):
    by = [y[i] for i in idx[i * 10: (i+1)*10]]
    bidx = idx[i * 10: (i + 1)*10]
    print(bidx)
    print(by)

a = torch.randn(10 ,4)
print(a)
a_np = a.numpy()
print(torch.from_numpy(np.delete(a_np, [2, 5], axis=0)))
print(a)

a = torch.Tensor([1, 2, 3])
for i in range(3):
    print(a[i])
a = [1 ,2 ,3]
# print(torch.arange(0, 5))

# a += [4, 5, 6]
# a.extend([4, 5, 6])
a.append([4, 5, 6])
print(a)

for index, i in enumerate(a, 0):
    print(index, i)

t = torch.randn((2, 1, 3))
print(t)
print(t.shape)
print(torch.squeeze(t))