import math
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal
from torch.nn import MultiheadAttention
from torchrl.modules import NoisyLinear

import config.config_constants as cfg
import config.config_objects as cfo
from actor import TorchActorModule
from custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN


# https://discuss.pytorch.org/t/dropout-in-lstm-during-eval-mode/120177
def gru(input_size, rnn_size, rnn_len, dropout: float = 0.0):
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

    # Start with the input layer size
    layers = [nn.Linear(dim_obs, sizes[0]), activation()]

    # Create each layer and add to the list
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())

    # Build the sequential model
    model = nn.Sequential(*layers)

    return model


class QRCNNSophy(nn.Module):
    # default value for each argument must be the same in every class in this file!!!
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

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        print(f"Observation dims in critic: {dim_obs}")
        dim_act = action_space.shape[0]
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self.mlp_api = mlp(mlp_branch_sizes[:-1], dim_obs, activation)

        # self.attentionAPI = MultiheadAttention(embed_dim=..., num_heads=2, batch_first=True)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-2])

        # self.rnn_block_api = gru(
        #     mlp_branch_sizes[-1] + dim_act,
        #     rnn_sizes[0],
        #     rnn_lens[0]
        # )
        self.mlp_act = mlp([mlp_branch_sizes[-1]], mlp_branch_sizes[-2] + dim_act, activation)

        self.attentionRNN = MultiheadAttention(
            embed_dim=mlp_branch_sizes[-1], num_heads=2, batch_first=True
        )

        if cfg.NOISY_LINEAR_CRITIC:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], self.num_quantiles)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

        self.h0 = None
        self.c0 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)

    def forward(self, observation, act, save_hidden=False):
        # self.rnn_block_api.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        # if not save_hidden or self.h0 is None or self.c0 is None:
        #     device = observation[0].device
        #     h0 = Variable(
        #         torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device)
        #     )
        #     # c0 = Variable(
        #     #     torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device)
        #     # )
        #
        # else:
        #     h0 = self.h0
        #     # c0 = self.c0

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        # mlp_api_out, _ = self.attentionAPI(mlp_api_out, mlp_api_out, mlp_api_out)

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        cat_mlp_api_act_out = torch.cat([mlp_api_out, act], dim=-1)

        # rnn_block_api_out, (h0, c0) = self.rnn_block_api(cat_mlp_api_act_out, (h0, c0))

        # rnn_block_api_out, h0 = self.rnn_block_api(cat_mlp_api_act_out, h0)

        mlp_api_out = self.mlp_act(cat_mlp_api_act_out)

        attn_output, _ = self.attentionRNN(mlp_api_out, mlp_api_out, mlp_api_out)

        model_out = self.model_out(attn_output)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        # if save_hidden:
        #     self.h0 = h0
        # self.c0 = c0

        return torch.squeeze(model_out, -1)


class SquashedActorSophy(TorchActorModule):
    # default value for each argument must be the same in every class in this file!!!
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

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        # print(f"Observation dims critic: {dim_obs}")
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        # self.attentionAPI = MultiheadAttention(embed_dim=..., num_heads=2, batch_first=True)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        # self.rnn_block_api = gru(
        #     mlp_branch_sizes[-1],
        #     rnn_sizes[0],
        #     rnn_lens[0]
        # )

        self.attentionRNN = MultiheadAttention(
            embed_dim=mlp_branch_sizes[-1], num_heads=2, batch_first=True
        )

        if cfg.NOISY_LINEAR_ACTOR:
            self.model_out = NoisyLinear(
                rnn_sizes[0], mlp_out_size, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], mlp_out_size)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.h0 = None
        self.c0 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        # self.rnn_block_api.flatten_parameters()
        # self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        # if not save_hidden or self.h0 is None or self.c0 is None:
        #     device = observation[0].device
        #     h0 = Variable(
        #         torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device)
        #     )
        # c0 = Variable(
        #     torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device)
        # )
        # else:
        #     h0 = self.h0
        # c0 = self.c0

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        # Pack the tensors in observation_except_img to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        # mlp_api_out, _ = self.attentionAPI(mlp_api_out, mlp_api_out, mlp_api_out)

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        # rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))
        # rnn_block_api_out, h0 = self.rnn_block_api(mlp_api_out, h0)

        attn_output, _ = self.attentionRNN(mlp_api_out, mlp_api_out, mlp_api_out)

        model_out = self.model_out(attn_output)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

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

        # if save_hidden:
        #     self.h0 = h0
        # self.c0 = c0

        return pi_action, logp_pi

    def act(self, obs: tuple, test=False):
        obs_seq = list(obs)
        # obs_seq = list(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(
                observation=obs_seq, test=test, with_logprob=False, save_hidden=True
            )
            return a.cpu().numpy()


class SophyActorCritic(nn.Module):
    # default value for each argument must be the same in every class in this file!!!
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
        self.actor = SquashedActorSophy(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNSophy(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q2 = QRCNNSophy(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
