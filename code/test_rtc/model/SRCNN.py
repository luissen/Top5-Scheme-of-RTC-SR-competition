"""
@project: mobile_sr_evaluation
@author: sfzhou
@file: SRCNN.py
@ide: PyCharm
@time: 2019/5/15 18:00

"""
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, scale_factor, nc=3):
        super(SRCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(nc, 64, 9, 1, 4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 3*(scale_factor ** 2), 5, 1, 2),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, input):

        x = self.feature(input)
        return x