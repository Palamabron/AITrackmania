import random

import numpy as np
import torch
from torch import nn

from tmrl.custom.models.image_input._impala_utils import init_kaiming
from tmrl.registry import MODELS


@MODELS.register("impala_cnn")
class CNNModule(nn.Module):
    def __init__(
        self,
        mlp_out_size: int = 256,
        activation=nn.LeakyReLU,
        seed: int = 42,
        img_height: int = 64,
        img_width: int = 64,
        img_hist_len: int = 4,
        cnn_filters: list[int] | None = None,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        cnn_filters = cnn_filters or [64, 64, 128, 128]

        self.img_height = img_height
        self.img_width = img_width
        self.img_hist_len = img_hist_len

        self.activation = activation()
        self.conv_groups = 2
        self.conv_blocks = nn.ModuleList()
        self.out_activation = nn.ReLU()
        h_out, w_out = img_height, img_width
        hist = img_hist_len
        filters = cnn_filters

        def calculate_output_size(h_w, kernel_size, stride, padding, pool_kernel=3, pool_stride=2):
            h, w = h_w
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1

            h = (h - pool_kernel) // pool_stride + 1
            w = (w - pool_kernel) // pool_stride + 1
            return h, w

        for i in range(len(filters)):
            last_index = -1 if i + 1 >= len(filters) else i + 1

            residual_block = nn.Sequential(
                nn.Conv2d(
                    filters[i] if i != 0 else hist,
                    filters[i],
                    kernel_size=3,
                    stride=1,
                    padding="same",
                    groups=1,
                ),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(
                    filters[i],
                    filters[i],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    groups=self.conv_groups,
                ),
                activation(),
                nn.Conv2d(
                    filters[i],
                    filters[i],
                    kernel_size=3,
                    stride=1,
                    padding="same",
                    groups=self.conv_groups,
                ),
                activation(),
                nn.Conv2d(
                    filters[i],
                    filters[i],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    groups=self.conv_groups,
                ),
                activation(),
                nn.Conv2d(
                    filters[i],
                    filters[last_index],
                    kernel_size=3,
                    stride=1,
                    padding="same",
                    groups=self.conv_groups,
                ),
                activation(),
            )

            self.conv_blocks.append(residual_block)
            h_out, w_out = calculate_output_size((h_out, w_out), kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        flat_features = self.flattendim((self.img_hist_len, self.img_height, self.img_width))
        self.mlp_out_size = mlp_out_size
        self.fc1 = nn.Linear(in_features=flat_features, out_features=mlp_out_size)
        self.initialize_weights()

    def flattendim(self, input_shape):
        temp_shape = list(input_shape)
        for seq in self.conv_blocks:
            for module in seq:  # type: ignore[attr-defined]
                if isinstance(module, nn.Conv2d):
                    cin, hin, win = temp_shape

                    kernel_size = (
                        module.kernel_size[0]
                        if isinstance(module.kernel_size, tuple)
                        else module.kernel_size
                    )
                    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                    padding = (
                        module.padding[0] if isinstance(module.padding, tuple) else module.padding
                    )

                    if isinstance(padding, str):
                        if padding == "same":
                            padding = kernel_size // 2
                        else:
                            raise ValueError(f"Unsupported padding value: {padding}")

                    hout = (hin + 2 * padding - (kernel_size - 1) - 1) // stride + 1
                    wout = (win + 2 * padding - (kernel_size - 1) - 1) // stride + 1
                    cout = module.out_channels

                    temp_shape = [cout, hout, wout]

                elif isinstance(module, nn.MaxPool2d):
                    cin, hin, win = temp_shape
                    kernel_size = (
                        module.kernel_size
                        if isinstance(module.kernel_size, int)
                        else module.kernel_size[0]
                    )
                    stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                    padding = (
                        module.padding if isinstance(module.padding, int) else module.padding[0]
                    )

                    hout = (hin + 2 * padding - (kernel_size - 1) - 1) // stride + 1
                    wout = (win + 2 * padding - (kernel_size - 1) - 1) // stride + 1

                    temp_shape = [cin, hout, wout]

        return int(np.prod(np.array(temp_shape)))

    def initialize_weights(self):
        for m in self.conv_blocks:
            if isinstance(m, torch.nn.Conv2d):
                init_kaiming(m)
        init_kaiming(self.fc1)

    def forward(self, x):
        x /= 255.0
        residual = None
        for i, layer in enumerate(self.conv_blocks):
            if i % 2 == 0:
                residual = x
            if (
                residual is not None
                and i > 0
                and (residual.size(2) == x.size(2) or residual.size(3) == x.size(3))
            ):
                x = x + residual
            x = layer(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out_activation(x)

        return x
