# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
# import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


def catch(sen_matrix, label_matrix, len_list):
        """
            out shape = batch word_len embedding_size
            label_matrix = batch 1 embedding_size
            len_list = true sentence_length
        """
        pos_list = []
        neg_list = []
        vector_list = []
        for i in range(len(len_list)):
            # print('len_list', len_list[i])
            # print(sen_matrix)
            # print(sen_matrix[i].shape[0])
            s_matrix = torch.index_select(sen_matrix[i], 0, torch.arange(0, int(len_list[i])))
            label_vector = label_matrix[i]
            s = s_matrix.mm(label_vector.t())
            # print(s.shape, len_list[i])
            value, index = torch.topk(s, k=4, largest=True, dim=0)
            index = torch.squeeze(index)
            s_value = torch.index_select(s_matrix, 0, torch.squeeze(index))
            index_list = index.tolist()
            for j in range(int(len_list[i])):
                if j in index_list:
                    continue
                else:
                    vector_list.append(s_matrix[j])
            neg_matrix = torch.stack(vector_list)
            final_matrix = s_value * value
            pos_vector = torch.sum(final_matrix, dim=0)
            neg_vector = torch.sum(neg_matrix, dim=0)
            pos_list.append(pos_vector)
            neg_list.append(neg_vector)
        pos_rep = torch.stack(pos_list)
        neg_rep = torch.stack(neg_list)
        return pos_rep, neg_rep


def confusion(sen, label, length_list, instance_l_list):
    pos_list = []
    neg_list = []
    for i in range(len(length_list)):
        sen_matrix = torch.index_select(sen[i], 0, torch.arange(0, int(length_list[i])))
        # print(label)
        word_l = torch.mm(sen_matrix, label.t())
        _, max_index = torch.max(word_l, dim=1)
        pos_index_list = []
        neg_index_list = []
        for index, item in enumerate(max_index, 0):
            if item == instance_l_list[i]:
                pos_index_list.append(index)
            else:
                neg_index_list.append(index)
        pos_index_t = torch.LongTensor(pos_index_list)
        neg_index_t = torch.LongTensor(neg_index_list)
        pos_m = torch.index_select(word_l, 0, pos_index_t)
        neg_m = torch.index_select(word_l, 0, neg_index_t)
        pos_v = torch.sum(pos_m, dim=0)
        neg_v = torch.sum(neg_m, dim=0)
        pos_list.append(pos_v)
        neg_list.append(neg_v)
    pos_rep = torch.stack(pos_list)
    neg_rep = torch.stack(neg_list)
    return pos_rep, neg_rep


class LSTM(nn.Module):

    def __init__(self, vocab_size, vector_len, hidden_size, num_layers, batch):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        self.embedding = nn.Embedding(vocab_size, vector_len)  # Embedding.weight.requires_grad default: True
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vector_len, hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=False, dropout=0.5)
        self.linear = nn.Linear(hidden_size, 6)
        self.label_word_linear = nn.Linear(vector_len, 6)
        self.llinear = nn.Linear(300, 100)
        # self.drop = nn.Dropout(0.5)
        # self.softmax = F.softmax()

    '''
        h_0 of shape (num_layers * num_directions, batch, hidden_size)
        c_0 of shape (num_layers * num_directions, batch, hidden_size)
    '''

    def init_hidden(self, bidirectional):
        # return (autograd.Variable(torch.randn(self.num_layers * 2, self.batch, self.hidden_size)),
        #         autograd.Variable(torch.randn(self.num_layers * 2, self.batch, self.hidden_size))) \
        #     if bidirectional else (autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_size)),
        #                            autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_size)))
        return (torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True),
                torch.zeros((self.num_layers * 2, self.batch, self.hidden_size), requires_grad=True)) \
            if bidirectional else (torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True),
                                   torch.zeros((self.num_layers, self.batch, self.hidden_size), requires_grad=True))

    def forward(self, data, multi_label=None):
        word_id = data[0]
        sen_len = data[1]
        # print(90*'*', self.embedding.weight.requires_grad)  # True
        word_vector = self.embedding(word_id)  # [batch_size, seq_len, embedding_size]
        # print('=============================',word_vector.shape)
        # print(word_vector.shape)
        label_matrix = None
        label_id = None
        if self.training:
            # confusion
            label_id = data[2]
            # label_word_id = torch.tensor(label_word_id)
            label_matrix = self.embedding(multi_label)  # shape [6, 1, dim]
            label_matrix = torch.squeeze(label_matrix)
        h0, c0 = self.init_hidden(False)
        # print(word_vector.shape)
        word_vector = rnn_utils.pack_padded_sequence(word_vector, lengths=sen_len, batch_first=True, enforce_sorted=False)
        # print(word_vector)
        out, (hn, cn) = self.lstm(word_vector, (h0, c0))
        # print(out.shape)
        # print(out)
        out, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
        # print(out.contiguous()[:, -1, :])
        # print(hn.shape)  # torch.Size([2, 10, 100])
        # print(out.shape)
        # print(sen_len)
        # print(out[:, -1, :].shape)  # [10, 300]
        # print('{}'.format(torch.equal(torch.Tensor(sen_len).long(), out_len)))
        # result = None
        # for i in range(len(sen_len)):
        #     if result is None:
        #         result = torch.unsqueeze(out[i][sen_len[i].long() - 1, :], dim=0)
        #     else:
        #         result = torch.cat((result, torch.unsqueeze(out[i][sen_len[i].long() - 1, :], dim=0)), dim=0)
        # 获得最后时间序列的vectors
        # print('out_len', out_len)
        out_len_f = (out_len - 1).view(out_len.shape[0], 1, -1)
        out_len_f = out_len_f.repeat(1, 1, self.hidden_size)
        out_f = torch.gather(out, 1, out_len_f)
        out_f = torch.squeeze(out_f, dim=1)
        # if self.training:  # model是在train还是eval可以用这个判断。
        #     label_out = self.label_word_linear(label_vector)
        #     out = self.linear(out)
        #     return out, label_out
        # else:
        out_f = self.linear(out_f)
        if self.training:
            label_matrix = self.llinear(label_matrix)
            # p_rep, n_rep = catch(out, label_matrix, out_len)
            p_rep, n_rep = confusion(out, label_matrix, out_len, label_id)
            # po_rep = self.linear(p_rep)
            # ne_rep = self.linear(n_rep)
            return out_f, out, label_matrix, out_len, label_id
        # out = self.linear(out)
        # out = self.drop(out)
        return out_f





