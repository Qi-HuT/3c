# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
def squash(x, axis=-1):
    x_squred = torch.square(x).sum(axis, keepdim=True)
    scale = torch.sqrt(x_squred) / (0.5 + x_squred)
    return scale * x


# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, vocab):  # in_channels = 1
#         super(ConvLayer, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, input_shape):
#         output1 = self.conv1(input_shape)
#         conv_result = self.relu(output1)
#         return conv_result


class LowerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, lower_layer_num, kernel_size):
        super(LowerLayer, self).__init__()
        # self.conv_group = nn.ModuleList([
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
        #     for _ in range(firstlayer_num)])
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * lower_layer_num,
                              kernel_size=kernel_size, stride=1, padding=0)
        self.out_channels = out_channels

    def forward(self, x):
        u = self.conv(x)
        # print(u.shape)
        batch_size = u.shape[0]
        return squash(u.reshape(batch_size, -1, self.out_channels), axis=-1)


class HigherLayer(nn.Module):
    def __init__(self, batch_size, lcap_num, lcap_dim, hcap_num, hcap_dim, iterations):
        super(HigherLayer, self).__init__()
        self.iterations = iterations
        self.lcap_dim = lcap_dim
        self.hcap_num = hcap_num
        self.hcap_dim = hcap_dim
        self.W = nn.Parameter(0.01 * torch.randn(batch_size, self.hcap_num, lcap_num, self.hcap_dim, self.lcap_dim),
                         requires_grad=True)

    def forward(self, u, lcap_num):
        batch_size = u.shape[0]
        u = u.unsqueeze(1).unsqueeze(4)
        # W = nn.Parameter(torch.randn(batch_size, self.hcap_num, lcap_num, self.hcap_dim, self.lcap_dim),
        #                  requires_grad=True)
        # print(W.shape)
        # print('W', W.shape)
        # print('u', u.shape)
        u_hat = torch.matmul(self.W, u)
        u_hat = u_hat.squeeze(-1)  # 去掉维度为1的

        b = torch.zeros(batch_size, self.hcap_num, lcap_num, 1).to(device)

        for i in range(self.iterations):
            c_ij = b.softmax(dim=1)
            s = (c_ij * u_hat).sum(dim=2)
            v = squash(s)
            if i < self.iterations - 1:
                # v.shape:(batch_size, out_caps, out_dim) ---> (batch_size, out_caps, out_dim, 1)
                # print('u->uv', u.shape)
                # print('v', v.unsqueeze(-1).shape)
                uv = torch.matmul(u_hat, v.unsqueeze(-1))
                b = uv
        v_norm = torch.norm(v, dim=-1)
        return v_norm


class FinalModel(nn.Module):
    def __init__(self, kernel_size, vocab_size):
        super(FinalModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])
        self.cnn = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(kernel_size, vocab_size[1]), stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.low_layer = LowerLayer(in_channels=256, out_channels=32, lower_layer_num=8, kernel_size=(kernel_size, 1))

        self.high_layer = HigherLayer(batch_size=10, lcap_num=368, lcap_dim=32, hcap_num=6, hcap_dim=16, iterations=3)

    def forward(self, data):

        word_id = self.embedding(data)
        data = word_id.unsqueeze(dim=1)
        # print("wordtovector:", data.shape)
        cnn_out = self.cnn(data)
        # print('vectorinconv', cnn_out.shape)
        cnn_out = self.relu(cnn_out)
        low_out = self.low_layer(cnn_out)
        lcap_num = low_out.shape[1]
        high_out = self.high_layer(low_out, lcap_num=lcap_num)
        return high_out


class ModelLoss(nn.Module):
    def __init__(self, upper, lower, lambd):
        super(ModelLoss, self).__init__()
        self.upper = upper
        self.lower = lower
        self.lambd = lambd

    def forward(self, v_matrix, label):
        left = (self.upper - v_matrix).relu() ** 2
        right = (v_matrix - self.lower).relu() ** 2
        loss = torch.sum(label * left) + self.lambd * torch.sum((1 - label) * right)
        return loss