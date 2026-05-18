from __future__ import annotations

import torch
from torch import nn


class BaseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: nn.Module | None = None,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.use_bn = use_bn
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv(torch.cat([avg, max_value], dim=1))


class BAM(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.sigmoid(self.channel(x))
        spatial = self.sigmoid(self.spatial(x))
        return x * (1.0 + channel * spatial)


class VGGBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.bam1 = BAM(128, 81)
        self.bam2 = BAM(256, 41)
        self.bam3 = BAM(512, 21)
        self.bam4 = BAM(512, 11)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1_1(image)
        x = self.conv1_2(x)
        x = self.pool(x)
        x = self.conv2_1(x)
        conv2 = self.bam1(self.conv2_2(x))

        x = self.pool(conv2)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3 = self.bam2(self.conv3_3(x))

        x = self.pool(conv3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        conv4 = self.bam3(self.conv4_3(x))

        x = self.pool(conv4)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        conv5 = self.bam4(self.conv5_3(x))
        return conv2, conv3, conv4, conv5


class DensityDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = BaseConv(1024, 512, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv2 = BaseConv(512, 256, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv3 = BaseConv(256, 256, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4 = BaseConv(512, 256, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv5 = BaseConv(256, 128, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv6 = BaseConv(128, 128, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv7 = BaseConv(256, 64, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv8 = BaseConv(64, 32, 3, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv9 = BaseConv(32, 32, 3, activation=nn.ReLU(inplace=True), use_bn=True)

    def forward(self, conv2: torch.Tensor, conv3: torch.Tensor, conv4: torch.Tensor, conv5: torch.Tensor) -> torch.Tensor:
        x = self.up(conv5)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.up(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv6(self.conv5(self.conv4(x)))
        x = self.up(x)
        x = torch.cat([x, conv2], dim=1)
        return self.conv9(self.conv8(self.conv7(x)))


class RiceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = VGGBackbone()
        self.amp = DensityDecoder()
        self.dmp = DensityDecoder()
        self.conv_att = BaseConv(32, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, activation=None, use_bn=False)

    def forward(self, image: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
        features = self.vgg(image)
        attention_feature = self.amp(*features)
        density_feature = self.dmp(*features)
        attention = self.conv_att(attention_feature)
        fused_feature = attention * density_feature
        density = self.conv_out(fused_feature)
        if return_features:
            return {
                "density": density,
                "attention": attention,
                "density_feature": fused_feature,
            }
        return density
