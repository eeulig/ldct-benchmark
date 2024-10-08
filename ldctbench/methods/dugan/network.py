import torch
import torch.nn as nn
import torch.nn.functional as F

from ldctbench.methods.redcnn.network import Model


def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(chan_out, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, kernel_size=1, stride=(2 if downsample else 1)
        )
        self.net = double_conv(input_channels, filters)
        self.down = (
            nn.Conv2d(filters, filters, kernel_size=4, padding=1, stride=2)
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        # Pad  if necessary
        diffY = res.size()[2] - x.size()[2]
        diffX = res.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv2d(input_channels // 2, out_channels, kernel_size=1)
        self.conv = double_conv(input_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, up):
        x = self.up(x)
        # Pad if necessary
        diffY = up.size()[2] - x.size()[2]
        diffX = up.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        p = self.conv(torch.cat((x, up), dim=1))
        sc = self.shortcut(x)
        return p + sc


class UNet(nn.Module):
    def __init__(
        self,
        repeat_num: int,
        use_tanh: bool = False,
        use_sigmoid: bool = False,
        skip_connection: bool = True,
        use_discriminator: bool = True,
        conv_dim: int = 64,
        in_channels: int = 1,
    ):
        """Unet discriminator

        Parameters
        ----------
        repeat_num : int
            Number of downsampling steps
        use_tanh : bool, optional
            Use tanh as final nonlinearity, by default False
        use_sigmoid : bool, optional
            Use sigmoid as final nonlinearity, by default False
        skip_connection : bool, optional
            Use skip connection, by default True
        use_discriminator : bool, optional
            Use encoder (at bottleneck) as additional output, by default True
        conv_dim : int, optional
            Number of features in first layer, by default 64
        in_channels : int, optional
            Number of channels of first layer, by default 1
        """
        super().__init__()
        self.use_tanh = use_tanh
        self.skip_connection = skip_connection
        self.use_discriminator = skip_connection
        self.use_sigmoid = use_sigmoid

        filters = [in_channels] + [
            min(conv_dim * (2**i), 512) for i in range(repeat_num + 1)
        ]
        filters[-1] = filters[-2]

        channel_in_out = list(zip(filters[:-1], filters[1:]))

        self.down_blocks = nn.ModuleList()

        for i, (in_channel, out_channel) in enumerate(channel_in_out):
            self.down_blocks.append(
                DownBlock(
                    in_channel, out_channel, downsample=(i != (len(channel_in_out) - 1))
                )
            )

        last_channel = filters[-1]
        if use_discriminator:
            self.to_logit = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(last_channel, 1),
            )

        self.up_blocks = nn.ModuleList(
            list(map(lambda c: UpBlock(c[1] * 2, c[0]), channel_in_out[:-1][::-1]))
        )

        self.conv = double_conv(last_channel, last_channel)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, input):
        x = input
        residuals = []

        for i in range(len(self.down_blocks)):
            x, unet_res = self.down_blocks[i](x)
            residuals.append(unet_res)

        bottom_x = self.conv(x) + x
        x = bottom_x
        for up_block, res in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)
        dec_out = self.conv_out(x)

        if self.use_discriminator:
            enc_out = self.to_logit(bottom_x)
            if self.use_sigmoid:
                dec_out = torch.sigmoid(dec_out)
                enc_out = torch.sigmoid(enc_out)
            return enc_out.squeeze(), dec_out

        if self.skip_connection:
            dec_out += input
        if self.use_tanh:
            dec_out = torch.tanh(dec_out)

        return dec_out
