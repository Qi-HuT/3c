# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
# import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

def catch(sen_matrix, label_matrix, len_list):
            '''
                out shape = batch word_len embedding_size
                label_matrix = batch 1 embedding_size
                len_list = true sentence_length
            '''
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
            neg_pre = torch.stack(neg_list)
            return pos_rep, neg_pre


class LSTM(nn.Module):

    def __init__(self, vocab_size, vector_len, hidden_size, num_layers, batch):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        self.embedding = nn.Embedding(vocab_size, vector_len)  # Embedding.weight.requires_grad default: True
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vector_len, hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=False, dropout=0.25)
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

    def forward(self, data):
            word_id = data[0]
            sen_len = data[1]
            # print(90*'*', self.embedding.weight.requires_grad)  # True
            word_vector = self.embedding(word_id)  # [batch_size, seq_len, embedding_size]
            # print('=============================',word_vector.shape)
            # print(word_vector.shape)
            if self.training:
                label_word_id = data[3]
                # label_word_id = torch.tensor(label_word_id)
                label_vector = self.embedding(label_word_id)
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
                label_vector = self.llinear(label_vector)
                l_rep, r_rep = catch(out, label_vector, out_len)
                l_rep = self.linear(l_rep)
                r_rep = self.linear(r_rep)
                return out_f, l_rep, r_rep
            # out = self.linear(out)
            # out = self.drop(out)
            return out_f





