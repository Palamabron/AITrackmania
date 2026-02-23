import math
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfo
from tmrl.actor import TorchActorModule
from tmrl.custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN


# https://discuss.pytorch.org/t/dropout-in-lstm-during-eval-mode/120177
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


# def mlp(sizes, dim_obs, activation=nn.ReLU):
#     layers = []
#     for j in range(len(sizes) - 1):
#         if j == 0:
#             layers += [nn.Linear(dim_obs, sizes[j + 1]), activation()]
#         else:
#             layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
#     return nn.Sequential(*layers)


def mlp(sizes, dim_obs, activation=nn.ReLU):
    """
    Build a neural network with the specified sizes, input dimension, and activation function.

    :param sizes: List of sizes for each linear layer.
    :param dim_obs: Size of the input.
    :param activation: Activation function to be used between layers.
    :return: Sequential model.
    """

    # Start with the input layer size
    layers = [nn.Linear(dim_obs, sizes[0]), activation()]

    # Create each layer and add to the list
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())

    # Build the sequential model
    model = nn.Sequential(*layers)

    return model


# def conv2d_out_dims(conv_layer, h_in, w_in):
#     if conv_layer.padding == "same":
#         return h_in, w_in
#     h_out = h_in + 2*conv_layer.padding[0] - conv_layer.dilation[0]*(...)
#     h_out = math.floor(h_out / conv_layer.stride[0] + 1)
#     w_out = w_in + 2*conv_layer.padding[1] - conv_layer.dilation[1]*(...)
#     w_out = math.floor(w_out / conv_layer.stride[1] + 1)
#     return h_out, w_out


def init_kaiming(layer):
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
    torch.nn.init.zeros_(layer.bias)


class CNNModule(nn.Module):
    def __init__(
        self, mlp_out_size: int = cfg.CNN_OUTPUT_SIZE, activation=nn.LeakyReLU, seed: int = cfg.SEED
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.activation = activation()
        self.conv_groups = 2
        self.conv_blocks = nn.ModuleList()
        self.out_activation = nn.ReLU()
        h_out, w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        hist = cfg.IMG_HIST_LEN
        filters = cfg.CNN_FILTERS

        def calculate_output_size(h_w, kernel_size, stride, padding, pool_kernel=3, pool_stride=2):
            h, w = h_w
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
            # Max Pool
            h = (h - pool_kernel) // pool_stride + 1
            w = (w - pool_kernel) // pool_stride + 1
            return h, w

        for i in range(len(filters)):
            if i + 1 >= len(filters):
                last_index = -1
            else:
                last_index = i + 1

            # Create a residual block
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

            # Add the residual block with a skip connection
            self.conv_blocks.append(residual_block)
            h_out, w_out = calculate_output_size((h_out, w_out), kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        # flat_features = h_out * w_out * filters[-1]
        flat_features = self.flattendim((cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH))
        self.mlp_out_size = mlp_out_size
        self.fc1 = nn.Linear(in_features=flat_features, out_features=mlp_out_size)
        self.initialize_weights()

    # def flattendim(self, input_shape):
    #     for sequential in self.conv_blocks:
    #         for module in sequential:
    #             name = module.__class__.__name__
    #             if name == 'Conv2d':
    #                 (cin, hin, win) = input_shape
    #                 xpad = (module.padding[0] if module.padding[0] != 's'
    #                         else module.kernel_size[0] - 1)
    #                 ypad = (module.padding[1] if module.padding[1] != 'a'
    #                         else module.kernel_size[1] - 1)
    #                 cout = module.out_channels
    #                 hout = int(np.floor(
    #                     (hin + 2 * xpad - module.dilation[0] * (module.kernel_size[0] - 1) - 1) /
    #                     module.stride[0] + 1))
    #                 wout = int(np.floor(
    #                     (win + 2 * ypad - module.dilation[1] * (module.kernel_size[1] - 1) - 1) /
    #                     module.stride[1] + 1))
    #                 input_shape = (cout, hout, wout)
    #             elif name == 'MaxPool2d':
    #                 (cin, hin, win) = input_shape
    #                 cout = cin
    #                 hout = int(np.floor(
    #                     (hin + 2*module.padding - module.dilation*(module.kernel_size-1)-1)
    #                     / module.stride + 1))
    #                 wout = int(np.floor(
    #                     (win + 2*module.padding - module.dilation*(module.kernel_size-1)-1)
    #                     / module.stride + 1))
    #                 input_shape = (cout, hout, wout)
    #
    #     return int(np.prod(np.array(input_shape)))

    def flattendim(self, input_shape):
        temp_shape = list(input_shape)  # Convert tuple to list for easy manipulation
        for seq in self.conv_blocks:
            for module in seq:
                if isinstance(module, nn.Conv2d):
                    cin, hin, win = temp_shape

                    # Ensure kernel_size, stride, and padding are integers
                    kernel_size = (
                        module.kernel_size[0]
                        if isinstance(module.kernel_size, tuple)
                        else module.kernel_size
                    )
                    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                    padding = (
                        module.padding[0] if isinstance(module.padding, tuple) else module.padding
                    )

                    # Check if padding is a string and handle it
                    if isinstance(padding, str):
                        if padding == "same":
                            padding = (
                                kernel_size // 2
                            )  # This is a common way to compute 'same' padding
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

                    temp_shape = [cin, hout, wout]  # channel count remains the same for MaxPool

        return int(np.prod(np.array(temp_shape)))  # Return the total number of features

    def initialize_weights(self):
        for m in self.conv_blocks:
            if isinstance(m, torch.nn.Conv2d):
                init_kaiming(m)
        init_kaiming(self.fc1)

    def forward(self, x):
        x /= 255.0
        i = 0
        residual = None
        for layer in self.conv_blocks:
            if i % 2 == 0:
                residual = x
            if (residual.size(2) == x.size(2) or residual.size(3) == x.size(3)) and i > 0:
                x = x + residual
            x = layer(x)
            i += 1

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out_activation(x)

        return x


class QRCNNQFunction(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0] + dim_act,
            rnn_sizes[1],
            rnn_lens[1],
            dropout=cfg.RNN_DROPOUT,
        )

        if cfg.NOISY_LINEAR_CRITIC:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], self.num_quantiles)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)
        # self.rnn_sizes[-1] += dim_act
        self.img_index = -3

    def forward(self, observation, act, save_hidden=False):
        self.rnn_block_api.flatten_parameters()
        self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[-3].float()
        if batch_size == 1:
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not cfg.GRAYSCALE:
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

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        # cat_layer_norm_act_out = torch.cat([layer_norm_api_out, act], dim=-1)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out, act], dim=-1)

        rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        if save_hidden:
            self.h0 = h0
            self.c0 = c0
            self.h1 = h1
            self.c1 = c1

        return torch.squeeze(model_out, -1)


class SquashedActorQRCNN(TorchActorModule):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0],
            rnn_sizes[1],
            rnn_lens[1],
            dropout=cfg.RNN_DROPOUT,
        )

        if cfg.NOISY_LINEAR_ACTOR:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], mlp_out_size)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

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
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)

            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not cfg.GRAYSCALE:
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

        # Pack the tensors in observation_except_img to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out], dim=-1)

        rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # pi_distribution = None
        # Pre-squash distribution and sample
        # if cfg.PI_DISTRIBUTION == "tanhnormal":
        #     pi_distribution = TanhNormal(mu, std)
        # else:
        pi_distribution = Normal(mu, std)

        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
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
        # obs_seq = list(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(
                observation=obs_seq, test=test, with_logprob=False, save_hidden=True
            )
            return a.cpu().numpy()


class QRCNNActorCritic(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # build policy and value functions
        self.actor = SquashedActorQRCNN(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q2 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
