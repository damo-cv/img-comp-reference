# coding=utf-8
"""
"""

import torch
from torch import nn
import torch.nn.functional as F
import random

from .ops import MaskedConv2d


class Conv2dUnfold(nn.Conv2d):
    def __init__(self, mask=True, *args, **kwargs):
        super(Conv2dUnfold, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.is_mask = mask
        if(mask):
            self.mask.fill_(1)
            self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x_unfold, h, w):
        if(self.is_mask):
            self.weight.data *= self.mask
        k = self.weight.size(2)
        out_unfold = x_unfold.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        out = F.fold(out_unfold, (h,w), (1,1))
        return out
        
    def forward_origin(self, x):
        self.weight.data *= self.mask
        return super(Conv2dUnfold, self).forward(x)


class SearchTransfer(nn.Module):
    def __init__(self, C, k=3, split=1):
        super().__init__()
        # modified
        # Mask Type 1
        mask = torch.ones((C // split, k, k))
        mask[:, k // 2, k // 2:] = 0
        mask[:, k // 2 + 1:, :] = 0
        mask_unfold = F.unfold(mask.unsqueeze(0), kernel_size=(k, k), padding=0)
        self.mask_unfold = torch.nn.Parameter(mask_unfold, requires_grad=False)
        self.k = k
        self.split = split

    def forward(self, y_hat, y_prob):
        k = self.k

        # Search
        unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k//2) * self.mask_unfold
        unfold = F.normalize(unfold, dim=1)  # [N, C*k*k, H*W]
        unfold_T = unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        R = torch.bmm(unfold_T, unfold) #[N, H*W, H*W]

        # Refer all when trainning, modified
        if(self.training):
            R = torch.triu(R, diagonal=1) + torch.tril(R, diagonal=-1)
        else:
            R = torch.triu(R, diagonal=1)

        R_star, R_star_arg = torch.max(R, dim=1) #[N, H*W]

        ### transfer
        y_hat_unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k//2)
        ref_unfold = self.bis(y_hat_unfold, 2, R_star_arg)  # [N, C*k*k, H*W]
        unfold_prob = F.unfold(y_prob, kernel_size=(1, 1), padding=0)
        U_unfold = self.bis(unfold_prob, 2, R_star_arg) # [N, 1, H*W]

        n, c, h, w = y_hat.shape
        S = R_star.view(n, 1, h, w)
        U = F.fold(U_unfold, output_size=(h,w), kernel_size=(1,1), padding=0)

        # Zero init the first pixel.
        if not self.training:
            S[:,:,0,0], U[:,:,0,0], ref_unfold[:,:,0], R_star_arg[:,0]  = 1e-8, 1e-8, 0., -1

        S = torch.clamp(S, min=1e-8, max=1.0)
        U = torch.clamp(U, min=1e-8, max=1.0)

        return S, U, ref_unfold, R_star_arg

    def forward_unidirectional(self, y_hat, y_prob):  # modified
        k = self.k

        # Search
        unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k//2) * self.mask_unfold
        unfold = F.normalize(unfold, dim=1)  # [N, C*k*k, H*W]
        unfold_T = unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        R = torch.bmm(unfold_T, unfold) #[N, H*W, H*W]

        # Refer all when trainning, modified
        R = torch.triu(R, diagonal=1)

        R_star, R_star_arg = torch.max(R, dim=1) #[N, H*W]

        ### transfer
        y_hat_unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k//2)
        ref_unfold = self.bis(y_hat_unfold, 2, R_star_arg)  # [N, C*k*k, H*W]
        unfold_prob = F.unfold(y_prob, kernel_size=(1, 1), padding=0)
        U_unfold = self.bis(unfold_prob, 2, R_star_arg) # [N, 1, H*W]

        n, c, h, w = y_hat.shape
        S = R_star.view(n, 1, h, w)
        U = F.fold(U_unfold, output_size=(h,w), kernel_size=(1,1), padding=0)

        # Zero init the first pixel.
        S[:,:,0,0], U[:,:,0,0], ref_unfold[:,:,0], R_star_arg[:,0]  = 1e-8, 1e-8, 0., -1

        S = torch.clamp(S, min=1e-8, max=1.0)
        U = torch.clamp(U, min=1e-8, max=1.0)

        return S, U, ref_unfold, R_star_arg

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


class RefAutoRegressiveModel(nn.Module):
    """
    Reference Based Auto Regressive Model implement with mask convolution.
    """
    def __init__(self, cin=128, chyper=128, cout=128, channels=128, sk=3, bias=True, random=True):
        """"""
        super().__init__()
        # Search and Reference module
        self.search = SearchTransfer(cin, k=sk)
        self.random = random

        # Local Context Model and Parameter Network
        self.mask_conv = MaskedConv2d('A', cin, cin*2, 5, 1, 2, bias=bias)
        self.conv_1x1_1 = torch.nn.Sequential(
            nn.Conv2d(cin*2, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, cout, 1, 1, 0, bias=bias)
        )

        # Global Reference Model and Parameter Network
        self.mask_conv_ref = Conv2dUnfold(True, cin, cin*2, sk, 1, sk//2, bias=bias)
        self.conv_1x1_2 = torch.nn.Sequential(
            nn.Conv2d(cin*2+cin*2, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, cout, 1, 1, 0, bias=bias)
        )

        # Parameter Network for Hyperprior
        self.conv_1x1_3 = torch.nn.Sequential(
            nn.Conv2d(cin*2+cin*2+chyper, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, cout, 1, 1, 0, bias=bias)
        )

    def forward(self, y_quant, z_feature, crit, warmup=False, soft=True):
        N, C, H, W = y_quant.shape

        # Local Context
        local_feature = self.mask_conv(y_quant)
        # Reference
        para1 = self.conv_1x1_1(local_feature.detach())
        y_prob1 = (- crit(y_quant, para1.clone())).exp_().mean(1, keepdim=True)
        S, U, ref_unfold, R = self.search(y_quant.detach(), y_prob1.detach())  # modified
        ref_feature = self.mask_conv_ref(ref_unfold, H, W)

        if self.training and self.random:
            random_mask(local_feature, ref_feature, z_feature)

        ## Cascade 1
        para1 = self.conv_1x1_1(local_feature)
        ## Cascade 2
        para2 = self.conv_1x1_2(torch.cat([local_feature, ref_feature], 1))
        para2 = para2.reshape(N, crit._num_params, C, 1, H, W)
        # soft-attention with Similarity 'S' and Uncertainty 'U'
        #para2[:,0,:,0,:,:] = (para2[:,0,:,0,:,:].clone().exp() * S * U + 1e-8).log()   #  Log-Sum-Exp trick
        para2[:,0,:,0,:,:] = para2[:,0,:,0,:,:].clone() + (S+1e-8).log() + (U+1e-8).log()  # Log-Sum-Exp trick modified
        # para2[:,0,:,0,:,:] = (para2[:,0,:,0,:,:].clone().exp() * (S+1e-8) * (U+1e-8)).log()   # modified
        para2 = para2.reshape(N, -1, H, W)
        ## Cascade 3
        para3 = self.conv_1x1_3(torch.cat([local_feature, ref_feature, z_feature], 1))

        return para1, para2, para3, S, U, R

    def random_mask(self, local, ref, hyper):
        _, _, h, w = local.shape
        for i in range(h):
            for j in range(w):
                prob = random.random()

                if prob < 0.25:  # context only
                    ref[:,:,i,j], hyper[:,:,i,j] = 0., 0.
                elif prob < 0.5:  # reference only
                    local[:,:,i,j], hyper[:,:,i,j] = 0., 0.
                elif prob < 0.75:  # hyperprior only
                    ref[:,:,i,j], local[:,:,i,j] = 0., 0.
                else:   # all enable
                    pass
