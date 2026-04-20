"""Validate canonical Hydra model presets against MainConfig schema."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tmrl.config.schema.main import MainConfig


def _compose_dict(overrides: list[str]) -> dict:
    cfg_dir = (Path(__file__).resolve().parents[1] / "tmrl" / "config" / "defaults").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out


PRESETS = [
    "vanilla_cnn_actor_critic",
    "vanilla_color_cnn_actor_critic",
    "sophy_actor_critic",
    "sophy_residual_actor_critic",
    "mlp_actor_critic",
    "residual_mlp_actor_critic",
    "redq_mlp_actor_critic",
    "rnn_actor_critic",
    "effnet_actor_critic",
]

DISCRETE_ONLY_PRESETS = {"vanilla_cnn_actor_critic", "vanilla_color_cnn_actor_critic"}

ALL_ALGORITHMS = ["sac", "iqn", "redqsac", "tqc", "sdsac"]
DISCRETE_ALGORITHMS = {"iqn", "sdsac"}


@pytest.mark.filterwarnings("ignore:IQN uses only:UserWarning")
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
@pytest.mark.parametrize("preset", PRESETS)
def test_algorithm_model_matrix_composes(algorithm: str, preset: str):
    """Hydra composes and Pydantic validates for every supported (algorithm, model) pair.

    Discrete algorithms (IQN, SDSAC) paired with continuous-only image presets
    are expected to be rejected by the schema validator.
    """
    raw = _compose_dict([f"algorithm={algorithm}", f"model={preset}"])

    should_reject_discrete = algorithm in DISCRETE_ALGORITHMS and preset in DISCRETE_ONLY_PRESETS

    if should_reject_discrete:
        with pytest.raises(ValueError, match="discrete-action-capable"):
            MainConfig.model_validate(raw)
        return

    main_cfg = MainConfig.model_validate(raw)
    assert main_cfg.algorithm is not None
    assert main_cfg.model is not None
