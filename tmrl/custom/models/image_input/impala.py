import math
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.model_constants import LOG_STD_MAX, LOG_STD_MIN
from tmrl.registry import MODELS


def gru(input_size, rnn_size, rnn_len, dropout: float = 0.1):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru_layers = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_rnn_layers,
        bias=True,
        batch_first=True,
        dropout=dropout,
        bidirectional=False,
    )

    return gru_layers


def lstm(input_size, rnn_size, rnn_len, dropout: float = 0.0):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    lstm_layers = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_rnn_layers,
        bias=True,
        batch_first=True,
        dropout=dropout,
        bidirectional=False,
    )

    return lstm_layers


def mlp(sizes, dim_obs, activation=nn.ReLU):
    """
    Build a neural network with the specified sizes, input dimension, and activation function.

    :param sizes: List of sizes for each linear layer.
    :param dim_obs: Size of the input.
    :param activation: Activation function to be used between layers.
    :return: Sequential model.
    """

    layers = [nn.Linear(dim_obs, sizes[0]), activation()]

    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())

    model = nn.Sequential(*layers)

    return model


def init_kaiming(layer):
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
    torch.nn.init.zeros_(layer.bias)


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
            for module in seq:
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


@MODELS.register("impala_qr_critic")
class QRCNNQFunction(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
        quantiles_number: int = 1,
        api_layernorm: bool = False,
        mlp_layernorm: bool = False,
        rnn_dropout: float = 0.0,
        noisy_linear_critic: bool = False,
        output_dropout: float = 0.0,
        grayscale: bool = False,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.grayscale = grayscale
        self.api_layernorm = api_layernorm
        self.mlp_layernorm = mlp_layernorm
        self.output_dropout_rate = output_dropout

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        self.num_quantiles = quantiles_number

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0] + dim_act,
            rnn_sizes[1],
            rnn_lens[1],
            dropout=rnn_dropout,
        )

        if noisy_linear_critic:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], self.num_quantiles)

        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)
        self.img_index = -3

    def forward(self, observation, act, save_hidden=False):
        self.rnn_block_api.flatten_parameters()
        self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[-3].float()
        if batch_size == 1:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            conv_branch_out = self.cnn_module(cnn_branch_input)

        if (
            not save_hidden
            or self.h0 is None
            or self.c0 is None
            or self.h1 is None
            or self.c1 is None
        ):
            device = observation[0].device
            h0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            c0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            h1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
            c1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
        else:
            h0 = self.h0
            c0 = self.c0
            h1 = self.h1
            c1 = self.c1

        observation[-3] = conv_branch_out

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        observation_except_img = observation[: self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1) :]

        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self.api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self.mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out, act], dim=-1)

        _rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if self.output_dropout_rate > 0.0:
            model_out = self.dropout(model_out)

        if save_hidden:
            self.h0 = h0
            self.c0 = c0
            self.h1 = h1
            self.c1 = c1

        return torch.squeeze(model_out, -1)


@MODELS.register("impala_qr_actor")
class SquashedActorQRCNN(TorchActorModule):
    """Squashed Gaussian policy over QRCNN features; default args must match config."""

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
        api_layernorm: bool = False,
        mlp_layernorm: bool = False,
        rnn_dropout: float = 0.0,
        noisy_linear_actor: bool = False,
        output_dropout: float = 0.0,
        grayscale: bool = False,
    ):
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.grayscale = grayscale
        self.api_layernorm = api_layernorm
        self.mlp_layernorm = mlp_layernorm
        self.output_dropout_rate = output_dropout

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0],
            rnn_sizes[1],
            rnn_lens[1],
            dropout=rnn_dropout,
        )

        if noisy_linear_actor:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], mlp_out_size)

        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)
        self.img_index = -3

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        self.rnn_block_api.flatten_parameters()
        self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[-3].float()
        if batch_size == 1:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)

            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)

            conv_branch_out = self.cnn_module(cnn_branch_input)

        if (
            not save_hidden
            or self.h0 is None
            or self.c0 is None
            or self.h1 is None
            or self.c1 is None
        ):
            device = observation[0].device
            h0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            c0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            h1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
            c1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
        else:
            h0 = self.h0
            c0 = self.c0
            h1 = self.h1
            c1 = self.c1

        observation[-3] = conv_branch_out

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        observation_except_img = observation[: self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1) :]

        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self.api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self.mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out], dim=-1)

        _rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if self.output_dropout_rate > 0.0:
            model_out = self.dropout(model_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - functional.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h0 = h0
            self.c0 = c0
            self.h1 = h1
            self.c1 = c1

        return pi_action, logp_pi

    def act(self, obs: tuple, test=False):
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(
                observation=obs_seq, test=test, with_logprob=False, save_hidden=True
            )
            return a.cpu().numpy()


@MODELS.register("impala_ac")
class QRCNNActorCritic(nn.Module):
    """QRCNN actor-critic; default constructor args must match config."""

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.actor = SquashedActorQRCNN(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q2 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
