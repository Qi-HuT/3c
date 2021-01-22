# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, vocab_size, vector_len, hidden_size, num_layers, batch):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.batch = batch
        self.embedding = nn.Embedding(vocab_size, vector_len)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vector_len, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, 5)
        self.softmax = F.softmax()

    '''
        h_0 of shape (num_layers * num_directions, batch, hidden_size)
        c_0 of shape (num_layers * num_directions, batch, hidden_size)
    '''

    def init_hidden(self, bidirectional):
        return (autograd.Variable(torch.randn(self.num_layers * 2, self.batch, self.hidden_size)),
                autograd.Variable(torch.randn(self.num_layers * 2, self.batch, self.hidden_size))) \
            if bidirectional else (autograd.Variable(torch.randn(self.num_layers, self.batch, self.hidden_size * 10)),
                                   autograd.Variable(torch.randn(self.num_layers, self.batch, self.hidden_size * 10)))

    def forward(self, data):
            pass

