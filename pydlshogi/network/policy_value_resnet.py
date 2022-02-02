import torch
import torch.nn as nn
import torch.nn.functional as F

from pydlshogi.common import *

in_ch = 104
ch = 256
fcl = 256

class Block(nn.Module):
    """ResNet Block

    input: torch.Size([batch_size, 104, 9, 9])
    output: torch.Size([batch_size, 104, 9, 9])

    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        """

        Args:
            x (torch.Size([batch_size, 104, 9, 9])): 入力

        Returns:
            torch.Size([batch_size, 104, 9, 9]): 出力

        """
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = self.bn2(self.conv2(h1))
        return F.relu(x + h2)

class PolicyValueResnet(nn.Module):
    """マルチタスク(方策, 価値)ResNet

    input: torch.Size([batch_size, 104, 9, 9]) 入力特徴量
    output: tuple(2)
        0: torch.Size([batch_size, MOVE_DIRECTION_LABEL_NUM]) 指し手の確率分布
        1: torch.Size([batch_size, 1]) 勝率(価値)

    """
    def __init__(self, block_num=10):
        """

        Args:
            block_num(int): Block数

        Returns:
            tuple(2):
                0: torch.Size([batch_size, MOVE_DIRECTION_LABEL_NUM]) 指し手の確率分布
                1: torch.Size([batch_size, 1]) 勝率(価値)

        """
        super().__init__()
        self.l1 = nn.Conv2d(in_channels = in_ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.blocks = nn.ModuleList([Block() for i in range(block_num)])
        # policy network
        self.lpolicy = nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1, bias=False)
        self.lpolicy_bias = nn.Parameter(1 / (ch * 2 * 3) * torch.rand(9 * 9 * MOVE_DIRECTION_LABEL_NUM))
        # value network
        self.lvalue1 = nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1)
        self.lvalue1_bn = nn.BatchNorm2d(MOVE_DIRECTION_LABEL_NUM)
        self.lvalue2 = nn.Linear(9 * 9 * MOVE_DIRECTION_LABEL_NUM, fcl)
        self.lvalue3 = nn.Linear(fcl, 1)

    def forward(self, x):
        """

        Args:
            x (torch.Size([batch_size, 104, 9, 9])): 入力特徴量

        Returns:
            tuple(2):
                0: torch.Size([batch_size, MOVE_DIRECTION_LABEL_NUM]) 指し手の確率分布
                1: torch.Size([batch_size, 1]) 勝率(価値)

        """
        x = F.relu(self.l1(x))
        for i, block in enumerate(self.blocks):
            x = block(x)
        # policy network
        policy = self.lpolicy(x)
        policy = torch.reshape(policy, (-1, 9 * 9 * MOVE_DIRECTION_LABEL_NUM))
        policy += self.lpolicy_bias
        # value network
        value = F.relu(self.lvalue1_bn(self.lvalue1(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.lvalue2(value))
        value = self.lvalue3(value)

        return policy, value
