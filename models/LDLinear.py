import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from typing import  Optional

'''
LDLinear: Replace the Moving Average Kernel (MOV) of DLinear with a Learnable Decomposition Module (LD), which is proposed in this paper:
https://openreview.net/forum?id=87CYNyCGOo
'''
class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.LD = LD(kernel_size=kernel_size)
        # self.decompsition = series_decomp(kernel_size)
        self.channels = configs.enc_in

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        trend_init = self.LD(x)
        seasonal_init = x - trend_init
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]