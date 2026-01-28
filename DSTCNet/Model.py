import math
from collections import OrderedDict
from thop import profile
import torch
import torch.nn as nn
import numpy as np
from scipy import signal

fs = 250


class Conv2dBlockELU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=(0, 0), dropout=0.0, dilation=(1, 1), groups=1):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.conv(x)


class SingleScaleBlock(nn.Module):
    def __init__(self, num_channel=10, filters=8, kernel_window=63):
        super().__init__()
        self.conv_time = Conv2dBlockELU(in_ch=filters, out_ch=filters,
                                        kernel_size=(1, kernel_window), padding=(0, int(kernel_window / 2)), dropout=0.3)
        self.conv_chan = Conv2dBlockELU(in_ch=filters, out_ch=filters, kernel_size=(num_channel, 1), dropout=0.1)

        self.avgpool = nn.AvgPool2d((1, 2))

    def forward(self, x):
        x = self.conv_time(x)  # [2, 8, 11, 250]
        x = self.conv_chan(x)  # [2, 8, 1, 250]
        x = self.avgpool(x)
        return x


class ECA_Block(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super().__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1

        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = x * inputs
        return outputs


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        # 可以整合一下重新写一个
        self.net = nn.Sequential(self.conv1, self.bn1, self.chomp1, self.elu1, self.dropout1,
                                 self.conv2, self.bn1, self.chomp2, self.elu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ChannelRecombinationModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, out_channel):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelRecombinationModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, out_channel),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
        )

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x1 = self.pointwise_conv(x)

        # 乘积获得结果
        x2 = Mc * x1

        return x2





class DSTCNet(nn.Module):

    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=48, kernel_window_global=25,               ##local=5,global=25, filters_n1=48
                 kernel_window_local=5, kernel_window=16):
        super().__init__()
        filters = [filters_n1, filters_n1 * 2]
        out_len = signal_length // 8

        self.global_pointwise = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=1, bias=False)
        self.global_block = SingleScaleBlock(num_channel, filters[0], kernel_window_global)     #############  加dropout  ###########

        self.local_pointwise = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=1, bias=False)
        self.local_block = SingleScaleBlock(num_channel, filters[0], kernel_window_local)

        self.atten = ChannelRecombinationModul(in_channel=filters[0] * 2, out_channel=filters[1])

        self.tcn1 = TemporalConvNet(filters[1], num_channels=[filters[1]] * 2, kernel_size=kernel_window, dropout=0.25)
        self.avgpool1 = nn.AvgPool2d((1, 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[1],
                kernel_size=(1, 3),
                bias=False,
                padding=(0, 1),
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ELU(),
            nn.Dropout(p=0.3)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[1],
                kernel_size=(1, 15),
                groups=filters[1],
                bias=False,
                padding=(0, 7)
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[1],
                kernel_size=(1, 5),
                bias=False,
                padding=(0, 2),
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )

        self.atten2 = ChannelRecombinationModul(in_channel=filters[1] * 2, out_channel=filters[1])

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(filters[1] * out_len, out_features=num_classes)

    def forward(self, x):  # [batchsize, 3, 11, 250]
        x1 = self.global_pointwise(x)
        x1 = self.global_block(x1)
        x2 = self.local_pointwise(x)
        x2 = self.local_block(x2)
        x = torch.cat((x1, x2), 1)

        x = self.atten(x)

        x3 = torch.squeeze(x)
        x3 = self.tcn1(x3)
        x3 = torch.unsqueeze(x3, 2)
        x3 = self.avgpool1(x3)

        x4 = self.conv1(x)
        x4 = self.depthwise_conv(x4)
        x4 = self.conv2(x4)
        x = torch.cat((x3, x4), 1)

        x = self.atten2(x)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def Filterbank(x, sampling, filterIdx):
    # x: time signal, np array with format (electrodes,data)
    # sampling: sampling frequency
    # filterIdx: filter index

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Nq = sampling / 2
    Wp = [passband[filterIdx] / Nq, 90 / Nq]
    Ws = [stopband[filterIdx] / Nq, 100 / Nq]
    [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
    [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')
    y = np.zeros(x.shape)
    channels = x.shape[0]
    for c in range(channels):
        y[c, :] = signal.filtfilt(B, A, x[c, :], padtype='odd', padlen=3 * (max(len(B), len(A)) - 1), axis=-1)
    return y