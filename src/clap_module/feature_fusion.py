'''
Feature Fusion for Varible-Length Data Processing
AFF/iAFF is referred and modified from https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
According to the paper: Yimian Dai et al, Attentional Feature Fusion, IEEE Winter Conference on Applications of Computer
 Vision, WACV 2021
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAF(nn.Module):
    """直接相加 DirectAddFuse
    """

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    """多特征融合 iAFF
    """

    def __init__(self, channels=64, r=4, type='2D'):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            # 第二次全局注意力
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            # 第二次全局注意力
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
            xa = torch.cat([xa, xa], dim=0)
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
    """多特征融合 AFF
    """

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
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class AttentionPool1d(nn.Module):
    def __init__(
            self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim
            # torch.randn(spacial_dim, embed_dim) / embed_dim
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.permute(1, 0, 2)  # B*L*D -> L*B*D; NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]  # B*D

