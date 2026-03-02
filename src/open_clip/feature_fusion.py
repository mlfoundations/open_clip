'''
Feature Fusion for Varible-Length Data Processing
AFF/iAFF is referred and modified from https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
According to the paper: Yimian Dai et al, Attentional Feature Fusion, IEEE Winter Conference on Applications of Computer Vision, WACV 2021
'''

import torch
import torch.nn as nn


class DAF(nn.Module):
    '''
    Direct Addition Fusion (DirectAddFuse)
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    Multi-feature Fusion iAFF
    '''

    def __init__(self, channels=64, r=4, type='2D'):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            # Local attention
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # Global attention
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # Second local attention
            self.local_att2 = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            # Second global attention
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            # Local attention
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # Global attention
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # Second local attention
            self.local_att2 = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            # Second global attention
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f'the type is not supported'

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa,xa],dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class AFF(nn.Module):
    '''
    Multi-feature Fusion AFF
    '''

    def __init__(self, channels=64, r=4, type='2D'):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f'the type is not supported.'
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa,xa],dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo
