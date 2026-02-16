import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
