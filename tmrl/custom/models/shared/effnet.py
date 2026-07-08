"""EfficientNetV2 classes, factory functions, and FrozenEfficientNetEncoder."""

from math import sqrt
from typing import cast

import torch
import torch.nn as nn

from tmrl.custom.models.shared.nn_utils import SiLU, _make_divisible

# ---------------------------------------------------------------------------
# EfficientNet building blocks
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention layer."""

    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block (with optional SE)."""

    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


def conv_3x3_bn(inp, oup, stride):
    """3x3 conv + BN + SiLU."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


def conv_dw_3x3_bn(inp, oup, stride):
    """Depthwise 3x3 + pointwise 1x1 stem (fewer FLOPs than conv_3x3_bn)."""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        SiLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


def conv_1x1_bn(inp, oup):
    """1x1 conv + BN + SiLU."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


# ---------------------------------------------------------------------------
# EfficientNetV2
# ---------------------------------------------------------------------------


class EffNetV2(nn.Module):
    """EfficientNetV2-style CNN.

    Args:
        cfgs: Block config list; each entry is
            ``[expand_ratio, channels, num_blocks, stride, use_se]``.
        nb_channels_in: Input channels (default 3 for RGB).
        dim_output: Output embedding dimension.
        width_mult: Channel width multiplier.
        use_dw_stem: Use depthwise-separable first conv (faster, slightly fewer params).
    """

    def __init__(
        self,
        cfgs,
        nb_channels_in: int = 3,
        dim_output: int = 1,
        width_mult: float = 1.0,
        use_dw_stem: bool = False,
    ):
        super().__init__()
        self.cfgs = cfgs
        input_channel = _make_divisible(24 * width_mult, 8)
        stem = (
            conv_dw_3x3_bn(nb_channels_in, input_channel, 2)
            if use_dw_stem
            else conv_3x3_bn(nb_channels_in, input_channel, 2)
        )
        layers = [stem]
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(MBConv(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_xs(**kwargs) -> EffNetV2:
    """EfficientNetV2-XS: 8 blocks, ~5x faster than S."""
    cfgs = [
        [1, 16, 1, 1, 0],
        [4, 32, 2, 2, 0],
        [4, 48, 2, 2, 0],
        [4, 96, 3, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_s(**kwargs) -> EffNetV2:
    """EfficientNetV2-S."""
    cfgs = [
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs) -> EffNetV2:
    """EfficientNetV2-M."""
    cfgs = [
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs) -> EffNetV2:
    """EfficientNetV2-L."""
    cfgs = [
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs) -> EffNetV2:
    """EfficientNetV2-XL."""
    cfgs = [
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


_EFFNET_VARIANTS = {
    "xs": effnetv2_xs,
    "s": effnetv2_s,
}


class FrozenEfficientNetEncoder(nn.Module):
    """EfficientNetV2 image encoder — frozen (default) or trainable.

    frozen=True: all parameters have requires_grad=False; encoder stays in train
    mode so BatchNorm uses batch statistics (running stats are never calibrated
    when the encoder starts from random init).
    frozen=False: trains end-to-end with the rest of the model.

    Supported variants: "xs" (8 blocks, fast), "s" (40 blocks).
    use_dw_stem=True: depthwise-separable first conv.
    """

    def __init__(
        self,
        nb_channels_in: int = 4,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        frozen: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._frozen = frozen
        factory = _EFFNET_VARIANTS.get(variant)
        if factory is None:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}', choose from {list(_EFFNET_VARIANTS)}"
            )
        self._encoder = factory(
            nb_channels_in=nb_channels_in,
            dim_output=embed_dim,
            width_mult=width_mult,
            use_dw_stem=use_dw_stem,
        )
        if frozen:
            for p in self._encoder.parameters():
                p.requires_grad = False
            self._encoder.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self._encoder.train()
        else:
            self._encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.5:
            x = x / 255.0
        if self._frozen:
            with torch.no_grad():
                return cast(torch.Tensor, self._encoder(x))
        return cast(torch.Tensor, self._encoder(x))
