import torch
import math
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, num_filters=128):
        super(ResBlock, self).__init__()
        # Conv2d里面的num_filters num_fil
        self.conv1 = nn.Conv2d(num_filters, num_filters // 2, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_filters//2, num_filters//2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_filters//2, num_filters, kernel_size=1, stride=1)

    def forward(self, x):
        res = self.relu1(self.conv1(x))
        res = self.relu2(self.conv2(res))
        res = self.conv3(res)
        res += x
        return res


class Attention(nn.Module):
    def __init__(self, num_filters=128):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.trunk_ResBlock1 = ResBlock(num_filters)
        self.trunk_ResBlock2 = ResBlock(num_filters)
        self.trunk_ResBlock3 = ResBlock(num_filters)
        self.attention_ResBlock1 = ResBlock(num_filters)
        self.attention_ResBlock2 = ResBlock(num_filters)
        self.attention_ResBlock3 = ResBlock(num_filters)

    def forward(self, x):
        trunk_branch = self.trunk_ResBlock1(x)
        trunk_branch = self.trunk_ResBlock2(trunk_branch)
        trunk_branch = self.trunk_ResBlock3(trunk_branch)

        attention_branch = self.attention_ResBlock1(x)
        attention_branch = self.attention_ResBlock2(attention_branch)
        attention_branch = self.attention_ResBlock3(attention_branch)
        attention_branch = self.conv1(attention_branch)
        attention_branch = self.sigmoid(attention_branch)

        # torch.mul是对应位置点   论文中提到的乘就是点乘的意思     torch.mm是矩阵的乘法   然后进行相加 
        result = x + torch.mul(attention_branch, trunk_branch)
        return result
