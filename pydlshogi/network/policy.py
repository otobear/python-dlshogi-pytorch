import torch
import torch.nn as nn
import torch.nn.functional as F

from pydlshogi.common import *

in_ch = 104
ch = 256 # 192

class PolicyNetwork(nn.Module):
    """方策ネットワーク

    input: torch.Size([batch_size, 104, 9, 9]) 入力特徴量
    output: torch.Size([batch_size, MOVE_DIRECTION_LABEL_NUM]) 指し手の確率分布

    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(in_channels = in_ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l2 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l3 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l4 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l5 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l6 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l7 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l8 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l9 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l10 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l11 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l12 = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = 3, padding = 1)
        self.l13 = nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1, bias=False)
        self.l13_bias = nn.Parameter(1 / (ch * 2 * 3) * torch.rand(9 * 9 * MOVE_DIRECTION_LABEL_NUM))
        self.norm1 = nn.BatchNorm2d(ch)
        self.norm2 = nn.BatchNorm2d(ch)
        self.norm3 = nn.BatchNorm2d(ch)
        self.norm4 = nn.BatchNorm2d(ch)
        self.norm5 = nn.BatchNorm2d(ch)
        self.norm6 = nn.BatchNorm2d(ch)
        self.norm7 = nn.BatchNorm2d(ch)
        self.norm8 = nn.BatchNorm2d(ch)
        self.norm9 = nn.BatchNorm2d(ch)
        self.norm10 = nn.BatchNorm2d(ch)
        self.norm11 = nn.BatchNorm2d(ch)
        self.norm12 = nn.BatchNorm2d(ch)

    def forward(self, x):
        """

        Args:
            x (torch.Size([batch_size, 104, 9, 9])): 入力特徴量

        Returns:
            torch.Size([batch_size, MOVE_DIRECTION_LABEL_NUM]): 指し手の確率分布

        """
        # without batch normalization: x = F.relu(self.l1(x))
        x = F.relu(self.norm1(self.l1(x)))
        x = F.relu(self.norm2(self.l2(x)))
        x = F.relu(self.norm3(self.l3(x)))
        x = F.relu(self.norm4(self.l4(x)))
        x = F.relu(self.norm5(self.l5(x)))
        x = F.relu(self.norm6(self.l6(x)))
        x = F.relu(self.norm7(self.l7(x)))
        x = F.relu(self.norm8(self.l8(x)))
        x = F.relu(self.norm9(self.l9(x)))
        x = F.relu(self.norm10(self.l10(x)))
        x = F.relu(self.norm11(self.l11(x)))
        x = F.relu(self.norm12(self.l12(x)))
        x = self.l13(x)
        x = torch.reshape(x, (-1, 9 * 9 * MOVE_DIRECTION_LABEL_NUM))
        x = x + self.l13_bias

        return x
