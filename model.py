# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
# import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class LSTM(nn.Module):

    def __init__(self, vocab_size, vector_len, hidden_size, num_layers, batch):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        self.embedding = nn.Embedding(vocab_size, vector_len)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vector_len, hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=False, dropout=0.25)
        self.linear = nn.Linear(hidden_size, 6)
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
            word_vector = self.embedding(word_id)  # [batch_size, seq_len, embedding_size]
            # print(word_vector.shape)
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
            out_len = (out_len - 1).view(out_len.shape[0], 1, -1)
            out_len = out_len.repeat(1, 1, self.hidden_size)
            out = torch.gather(out, 1, out_len)
            out = torch.squeeze(out, dim=1)
            out = self.linear(out)
            # out = self.drop(out)
            return out





