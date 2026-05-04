"""Vanilla CNN actor-critic for game frame + TM scalar observations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import Conv2d

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    conv2d_out_dims,
    mlp,
    num_flat_features,
    squashed_logprob,
)


def rgb_to_grayscale(images):
    """Extract first channel from RGB observations (shape: batch x hist x H x W x 3)."""
    return images[:, :, :, :, 0]


class VanillaCNN(nn.Module):
    """4-layer CNN over stacked game frames combined with TM scalar observations.

    Args:
        q_net: If True, builds a Q-network that also accepts an action tensor.
        img_height: Input image height in pixels.
        img_width: Input image width in pixels.
        img_hist_len: Number of stacked frames (input channels).
    """

    def __init__(
        self,
        q_net: bool,
        img_height: int = 64,
        img_width: int = 64,
        img_hist_len: int = 4,
    ):
        super().__init__()
        self.q_net = q_net
        h, w = img_height, img_width

        self.conv1 = Conv2d(img_hist_len, 64, 8, stride=2)
        h, w = conv2d_out_dims(self.conv1, h, w)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        h, w = conv2d_out_dims(self.conv2, h, w)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        h, w = conv2d_out_dims(self.conv3, h, w)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        h, w = conv2d_out_dims(self.conv4, h, w)

        self.flat_features = self.conv4.out_channels * h * w
        # scalars: speed + gear + rpm (3) + act1 + act2 (2x3=6) [+ act (3) for Q]
        scalar_dim = 12 if q_net else 9
        mlp_sizes = [self.flat_features + scalar_dim, 256, 256] + ([1] if q_net else [])
        self.mlp = mlp(mlp_sizes, nn.ReLU)

    def forward(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        out = F.relu(self.conv1(images))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        flat = num_flat_features(out)
        assert flat == self.flat_features
        out = out.view(-1, flat)

        if self.q_net:
            out = torch.cat((speed, gear, rpm, out, act1, act2, act), -1)
        else:
            out = torch.cat((speed, gear, rpm, out, act1, act2), -1)
        return self.mlp(out)


class VanillaCNNActor(TorchActorModule):
    """SAC actor using a VanillaCNN feature extractor."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = action_space.high[0]

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()
        logp_pi = squashed_logprob(pi_dist, pi_action) if with_logprob else None
        pi_action = self.act_limit * torch.tanh(pi_action)
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class VanillaCNNQFunction(nn.Module):
    """Q-function using a VanillaCNN feature extractor."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        q = self.net((*obs, act))
        return torch.squeeze(q, -1)


class VanillaCNNActorCritic(nn.Module):
    """Actor-critic with a shared VanillaCNN backbone."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.actor = VanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Color variants — discard colour channels before processing
# ---------------------------------------------------------------------------


class RGBVanillaCNNActor(VanillaCNNActor):
    """VanillaCNNActor that converts colour images to grayscale before processing."""

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        return super().forward(
            (speed, gear, rpm, rgb_to_grayscale(images), act1, act2), test, with_logprob
        )


class RGBVanillaCNNQFunction(VanillaCNNQFunction):
    """VanillaCNNQFunction that converts colour images to grayscale before processing."""

    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        return super().forward((speed, gear, rpm, rgb_to_grayscale(images), act1, act2), act)


class RGBVanillaCNNActorCritic(VanillaCNNActorCritic):
    """VanillaCNNActorCritic with colour-to-grayscale preprocessing."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.actor = RGBVanillaCNNActor(observation_space, action_space)
        self.q1 = RGBVanillaCNNQFunction(observation_space, action_space)
        self.q2 = RGBVanillaCNNQFunction(observation_space, action_space)
