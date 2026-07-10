"""IMPALA-style CNN encoder registered as ``impala_cnn``.

The encoder stacks residual convolutional blocks with inter-block skip connections,
followed by a fully-connected projection to a fixed-size embedding.
"""

import random

import numpy as np
import torch
from torch import nn

from tmrl.custom.models.image_input._impala_utils import init_kaiming
from tmrl.registry import MODELS


@MODELS.register("impala_cnn")
class CNNModule(nn.Module):
    """IMPALA-inspired residual CNN that encodes a stack of game frames to a 1-D embedding.

    Each ``cnn_filters`` stage is a residual block with MaxPool downsampling followed by
    two grouped convolution pairs. Skip connections are added every two stages when the
    spatial dimensions still match. Weights are initialised with Kaiming-normal.

    Pairs with ``impala_qr_actor`` / ``impala_qr_critic`` / ``impala_ac`` in the registry.

    Args:
        mlp_out_size: Width of the final linear projection (embedding dimension).
        activation: Activation class used throughout the convolutional blocks.
        seed: Random seed applied to Python, NumPy, and PyTorch for reproducibility.
        img_height: Spatial height of the input image in pixels.
        img_width: Spatial width of the input image in pixels.
        img_hist_len: Number of stacked frames (input channels to the first conv).
        cnn_filters: Channel widths for each residual stage. Defaults to [64, 64, 128, 128].
    """

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
        """Construct the IMPALA CNN encoder and compute the flattened feature dimension.

        Builds each residual stage as an ``nn.Sequential`` stored in
        ``self.conv_blocks``, then uses :meth:`flattendim` to derive the number of
        flat features fed into ``self.fc1`` without running a dummy forward pass.

        Args:
            mlp_out_size: Output embedding dimension of the final linear layer.
            activation: Activation class instantiated inside each residual block.
            seed: Seed for Python, NumPy and PyTorch RNGs and cuDNN determinism flags.
            img_height: Input image height in pixels.
            img_width: Input image width in pixels.
            img_hist_len: Number of stacked frames (first-layer input channels).
            cnn_filters: Per-stage output channels. Defaults to [64, 64, 128, 128].
        """
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
            """Compute (h, w) after one conv + one max-pool step."""
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

    def flattendim(self, input_shape: tuple) -> int:
        """Trace tensor shapes through all conv/pool layers and return the flat feature count.

        Does not allocate any tensors; walks the ``conv_blocks`` module tree and
        applies the standard output-size formulae for ``nn.Conv2d`` and ``nn.MaxPool2d``,
        including ``padding="same"`` expansion.

        Args:
            input_shape: (C, H, W) of the encoder input before any convolution.

        Returns:
            Total number of scalar features after flattening the final feature map.
        """
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

    def initialize_weights(self) -> None:
        """Apply Kaiming-normal initialisation to all Conv2d layers and the final fc."""
        for m in self.conv_blocks:
            if isinstance(m, torch.nn.Conv2d):
                init_kaiming(m)
        init_kaiming(self.fc1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of stacked frames to a fixed-length embedding.

        Normalises pixel values to [0, 1], then passes through each residual stage.
        A skip connection is added every two stages when the spatial dimensions still
        match (checked per H and W independently), so the first stage never receives
        a residual addition.

        Args:
            x: Input tensor of shape (N, img_hist_len, H, W), pixel values in [0, 255].

        Returns:
            Embedding tensor of shape (N, mlp_out_size) after ReLU activation.
        """
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
