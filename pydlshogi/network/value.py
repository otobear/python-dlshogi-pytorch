import torch
import torch.nn as nn
import torch.nn.functional as F

from pydlshogi.common import *

in_ch = 104
ch = 256
fcl = 256

class ValueNetwork(nn.Module):
    """価値ネットワーク

    input: torch.Size([batch_size, 104, 9, 9]) 入力特徴量
    output: torch.Size([batch_size, 1]) 勝率(価値)

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
        self.l13_v = nn.Conv2d(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, kernel_size = 1)
        self.l14_v = nn.Linear(9 * 9 * MOVE_DIRECTION_LABEL_NUM, fcl)
        self.l15_v = nn.Linear(fcl, 1)

    def forward(self, x):
        """

        Args:
            x (torch.Size([batch_size, 104, 9, 9])): 入力特徴量

        Returns:
            torch.Size([batch_size, 1]): 勝率(価値)

        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = F.relu(self.l9(x))
        x = F.relu(self.l10(x))
        x = F.relu(self.l11(x))
        x = F.relu(self.l12(x))
        x = F.relu(self.l13_v(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.l14_v(x))
        x = self.l15_v(x)

        return x
