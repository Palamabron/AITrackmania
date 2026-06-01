"""Neural network layout: CNN/RNN/MLP, residual stacks, EfficientNet, track encoder."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class BaseModelConfig(BaseModel):
    """Policy and value network structure (independent of optimizer loop settings)."""

    model_config = ConfigDict(extra="forbid")

    discrete_action_compatible: ClassVar[bool] = True
    """True when the preset can back IQN / SDSAC (discrete worker policy).

    False for vanilla image SAC stacks.
    """

    noisy_linear_critic: bool = Field(
        default=False,
        description="Enable NoisyNet factorized Gaussian noise on critic linear layers.",
    )
    noisy_linear_actor: bool = Field(
        default=False,
        description="Enable NoisyNet factorized Gaussian noise on actor linear layers.",
    )
    output_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Dropout probability on heads right before action/value outputs.",
    )
    rnn_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Dropout inside recurrent cores when use_rnn is enabled.",
    )
    cnn_filters: list[PositiveInt] = Field(
        default_factory=lambda: [32, 64, 64, 64],
        description="Output channels per convolutional stage in the vanilla CNN encoder.",
    )
    cnn_output_size: PositiveInt = Field(
        default=256,
        description="Flattened dimension after the CNN trunk before fusion MLPs.",
    )
    rnn_lens: list[PositiveInt] = Field(
        default_factory=lambda: [1],
        description="Unroll lengths for truncated BPTT when training recurrent policies.",
    )
    rnn_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [64],
        description="Hidden state width for each stacked recurrent layer.",
    )
    api_mlp_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [256, 256],
        description="Widths of fully-connected layers processing non-image features.",
    )
    api_layernorm: bool = Field(
        default=True,
        description="Apply LayerNorm in the API (vector) MLP stack.",
    )
    mlp_layernorm: bool = Field(
        default=False,
        description="Apply LayerNorm inside residual MLP blocks.",
    )
    use_residual_mlp: bool = Field(
        default=False,
        description="Use residual MLP backbones for boundary lidar / vector policies (non-image).",
    )
    residual_mlp_hidden_dim: PositiveInt = Field(
        default=256,
        description="Hidden width inside each residual MLP block.",
    )
    residual_mlp_num_blocks: PositiveInt = Field(
        default=6,
        description="Default depth when actor/critic-specific depths are left at zero.",
    )
    residual_mlp_num_blocks_actor: int = Field(
        default=0,
        ge=0,
        description="Actor-only depth; zero means reuse residual_mlp_num_blocks.",
    )
    residual_mlp_num_blocks_critic: int = Field(
        default=0,
        ge=0,
        description="Critic-only depth; zero means reuse residual_mlp_num_blocks.",
    )
    use_sophy_residual_actor: bool = Field(
        default=False,
        description=(
            "TQCGrab (or similar) with **no camera**: use SophyResidualActorCritic / "
            "SquashedActorSophyResidual (residual MLP trunk) instead of classic SophyActorCritic. "
            "Ignored for boundary lidar geometry, IQN, and image-based pipelines — omit from "
            "local.yaml unless you run that specific vector-TQC path."
        ),
    )
    split_track_observation: bool = Field(
        default=True,
        description=(
            "When true, the first observation tuple element (discretized track polyline) is "
            "encoded with track_encoder (conv1d | gtn | spline_mlp); remaining elements are "
            "telemetry (speed, etc.) merged after projection. When false, all observation "
            "components are flattened into one MLP path (track_encoder unused)."
        ),
    )
    use_simbav2: bool = Field(
        default=False,
        description="Toggle SimbaV2-specific fusion and normalization code paths.",
    )
    track_encoder: str = Field(
        default="conv1d",
        description=(
            "Track feature encoder family: conv1d, gtn, spline_mlp (gnn kept as legacy alias)."
        ),
    )
    gnn_layers: PositiveInt = Field(
        default=3,
        description="Number of message-passing iterations in the GNN track encoder.",
    )
    gnn_hidden: PositiveInt = Field(
        default=64,
        description="Hidden dimension for GNN node/edge embeddings.",
    )
    binary_brake: bool = Field(
        default=False,
        description="Snap continuous brake output to {0,1} at the final policy layer.",
    )
    use_rnn: bool = Field(
        default=False,
        description="Insert a recurrent core between perception and heads (advanced).",
    )
    rnn_hidden_size: int = Field(
        default=0,
        ge=0,
        description="Recurrent hidden size; zero falls back to residual_mlp_hidden_dim.",
    )
    use_efficientnet: bool = Field(
        default=True,
        description="Prefer EfficientNet backbones when building image encoders.",
    )
    use_frozen_effnet: bool = Field(
        default=False,
        description="Freeze EfficientNet weights and train only adapters and heads.",
    )
    frozen_effnet_embed_dim: PositiveInt = Field(
        default=256,
        description="Projected embedding size after the frozen EfficientNet trunk.",
    )
    frozen_effnet_width_mult: Annotated[float, Field(gt=0.0)] = Field(
        default=0.5,
        description="Width multiplier controlling EfficientNet parameter count vs speed.",
    )
    frozen_effnet_variant: str = Field(
        default="xs",
        description="Symbolic EfficientNet variant identifier understood by the model builder.",
    )
    frozen_effnet_use_dw_stem: bool = Field(
        default=False,
        description="Use depthwise-separable stem convolutions in the frozen wrapper.",
    )
