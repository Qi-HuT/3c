# -*- coding: utf-8 -*-
from transformers import TransfoXLModel, TransfoXLTokenizer, TransfoXLConfig
import torch
import collections
import pandas as pd
import os
import torch
from torch import nn
import torch.optim as optim
import time
from torchtext.vocab import Vocab, Vectors
from sklearn.metrics import f1_score
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def ls_att(llist, slist, length):
    # print(llist.device)
    # print(slist.device)
    # print(length.device)
    # llist = llist.cpu()
    # slist = llist.cpu()
    # length = length.cpu()
    pos__list = []
    neg_list = []
    for i in range(len(length)):
        # 选择真的有用的几行
        sen_matrix = torch.index_select(slist[i], 0, torch.arange(0, length[i] - 1).to(device))
        # sen_matrix = torch.index_select(slist[i], 0, torch.arange(0, length[i] - 1))
        l_matrix = llist[i]
        dot_value = sen_matrix.mm(l_matrix.t())
        value, index = torch.topk(dot_value, k=5, dim=0, largest=True)
        index = torch.squeeze(index)
        select_value = torch.index_select(sen_matrix, 0, torch.squeeze(index))
        # sen_matrix_np = sen_matrix.numpy()
        # index_np = index.numpu()
        # neg_matrix = torch.from_numpy(np.delete(sen_matrix_np, index_np, axis=0))
        final_matrix = select_value * value
        pos_vector = torch.sum(final_matrix, dim=0)
        pos__list.append(pos_vector)
    pos_rep = torch.stack(pos__list).to(device)
    # representation = torch.stack(vector_list)
    return pos_rep


class TransformerGetFeatures(nn.Module):
    def __init__(self, vocab):
        super(TransformerGetFeatures, self).__init__()
        self.embedding = nn.Embedding(vocab[0], vocab[1])
        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        self.linear = nn.Linear(1024, 6)
        self.feature_linear = nn.Linear(1024, 300)
        # self.smi_loss = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
        self.smi_loss = nn.CosineEmbeddingLoss(reduction='mean', size_average=True, reduce=True)

    def forward(self, input_object=None, target=None, cro_loss_function=None, run_type=None, label=None):
        # input_object = self.tokenizer(batch_sentences, return_tensors='pt', is_split_into_words=True, padding=True,
        #                               return_length=True)
        input_ids = input_object['input_ids']
        batch_length = input_object['length']
        to_label_length = batch_length
        batch_length = (batch_length - 1).view(input_ids.shape[0], 1, -1)
        batch_length = batch_length.repeat(1, 1, 1024)
        # 如何获得想要的features
        # tensor_3d = torch.LongTensor([1, 2, 3, 4, 5, 6])
        # tensor_3d = tensor_3d.view(6, 1, -1)
        # cat = tensor_3d.repeat(1, 1, 6)
        # print(cat)
        # ten_list = torch.randn(6, 7, 6)
        # print(ten_list)
        # print(torch.gather(ten_list, 1, cat))
        transformer_output = self.model(input_ids)
        # transformer_output.last_hidden_state.shape = torch.Size([8, 122, 1024])
        # print(transformer_output.last_hidden_state.shape)
        # features = self.feature_linear(transformer_output.last_hidden_state)
        # label_out = [8, 1, 300]
        # label_out = self.embedding(label)
        # l_repre = ls_att(label_out, features, to_label_length)
        # print(l_repre.shape)  # torch.Size([8, 300])
        gather_output = torch.gather(transformer_output.last_hidden_state, 1, batch_length)
        squeeze_output = torch.squeeze(gather_output, dim=1)
        # print(squeeze_output.shape)
        linear_output = self.linear(squeeze_output)
        # print(linear_output.shape)
        sim_output = self.feature_linear(squeeze_output)
        # if self.training:
        #     print('train')
        if run_type == 'train':
            features = self.feature_linear(transformer_output.last_hidden_state)
            label_out = self.embedding(label)
            l_repre = ls_att(label_out, features, to_label_length)
            cro_loss = cro_loss_function(linear_output, target)
            one = torch.ones((10, 1), dtype=torch.float)
            s_loss = self.smi_loss(sim_output, l_repre, one)
            loss = cro_loss + s_loss
            return loss, linear_output
        else:
            return linear_output
