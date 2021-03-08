# -*- coding: utf-8 -*-
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, second_out_channels):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=second_out_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()

    def forward(self, input_shape):
        output1 = self.conv1(input_shape)
        output2 = self.conv2(output1)
        conv_result = self.relu(output2)
        return conv_result


class FirstLayer(nn.Module):
    def __init__(self):
        super(FirstLayer, self).__init__()


class SecondLayer(nn.Module):
    def __init__(self):
        super(SecondLayer, self).__init__()
