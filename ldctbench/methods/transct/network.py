from argparse import Namespace

import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class SpaceToDepth(nn.Module):
    """PyTorch implementation of TensorFlow's `tf.nn.space_to_depth` taken from [here](https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/12)."""

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(
            N, C, H // self.bs, self.bs, W // self.bs, self.bs
        )  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(
            N, C * (self.bs**2), H // self.bs, W // self.bs
        )  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthToSpace(nn.Module):
    """PyTorch implementation of TensorFlow's `tf.nn.depth_to_space` taken from [here](https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/12)."""

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(
            N, self.bs, self.bs, C // (self.bs**2), H, W
        )  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(
            N, C // (self.bs**2), H * self.bs, W * self.bs
        )  # (N, C//bs^2, H * bs, W * bs)
        return x


class EncodeLayer(nn.Module):
    def __init__(self, n_ch, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=n_ch, num_heads=n_heads, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(n_ch, 8 * n_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(8 * n_ch, n_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x1, _ = self.mha(query=x, key=x, value=x)
        x = x1 + x
        x_ff = self.ff(x)
        return x + x_ff


class DecodeLayer(nn.Module):
    def __init__(self, n_ch, n_heads=8):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(
            embed_dim=n_ch, num_heads=n_heads, batch_first=True
        )
        self.mha2 = nn.MultiheadAttention(
            embed_dim=n_ch, num_heads=n_heads, kdim=256, vdim=256, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(n_ch, 8 * n_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(8 * n_ch, n_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x, memory):
        x1, _ = self.mha1(query=x, key=x, value=x)
        x = x1 + x
        x2, _ = self.mha2(query=x, key=memory, value=memory)
        x = x2 + x
        x3 = self.ff(x)
        return x3 + x


class ConvEncoderLF(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.lc1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.transformer_endoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = self.base_encoder(x)
        x_l_c1 = self.lc1(x)
        x_l_c2 = self.lc2(x_l_c1)
        x_att = self.transformer_endoder(x)
        return x_l_c1, x_l_c2, x_att


class Combine(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.depth2space1 = DepthToSpace(block_size=2)
        self.depth2space2 = DepthToSpace(block_size=8)

    def forward(self, x, x_l_c1, x_l_c2):
        x1 = x + x_l_c2
        x2 = self.block1(x1)
        x3 = self.depth2space1(x1 + x2)

        x3 = x3 + x_l_c1
        x4 = self.block2(x3)
        out = self.depth2space2(x3 + x4)
        return out


class Model(nn.Module):
    """Implements the TransCT model.

    Translated from the Tensorflow implementation provided by the authors [here](https://github.com/zzc623/TransCT).
    """

    def __init__(self, args: Namespace):
        """Init function

        Parameters
        ----------
        args : Namespace
            Command line arguments passed to the model.
        """
        super().__init__()
        self.gaussian = GaussianBlur(kernel_size=11, sigma=1.5)
        self.conv_encoder_lf = ConvEncoderLF()
        self.space2depth = SpaceToDepth(block_size=16)
        self.conv_encoder_hf = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.memory = nn.Sequential(
            EncodeLayer(n_ch=256), EncodeLayer(n_ch=256), EncodeLayer(n_ch=256)
        )
        self.decoder = nn.ModuleList(
            [DecodeLayer(n_ch=1024), DecodeLayer(n_ch=1024), DecodeLayer(n_ch=1024)]
        )
        self.combine = Combine()

    def forward(self, x):
        # Gaussian filter to split into hf and lf part
        x_l = self.gaussian(x)
        x_h = x - x_l

        # LF Part (top in Fig. 1)
        x_l_c1, x_l_c2, x_att = self.conv_encoder_lf(x_l)
        x_att_flat = x_att.flatten(-2)  # b, 256, 16x16
        memory = self.memory(x_att_flat)

        # HR part (bottom in Fig. 1)
        x_h = self.space2depth(x_h)
        x_h = self.conv_encoder_hf(x_h)
        x_h = x_h.flatten(-2)  # b, 256, 32x32
        for i in range(len(self.decoder)):
            x_h = self.decoder[i](x_h, memory)
        x_h = torch.reshape(x_h, (x_h.size(0), x_h.size(1), 32, 32))
        out = self.combine(x_h, x_l_c1, x_l_c2)
        return out
