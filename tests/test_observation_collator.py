from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from tmrl.builtins import GymnasiumObservationCollator
from tmrl.core.data import Transition


def _space() -> spaces.Dict:
    return spaces.Dict(
        {
            "continuous": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "nested": spaces.Tuple(
                (
                    spaces.Discrete(3),
                    spaces.MultiBinary(2),
                    spaces.MultiDiscrete([2, 4]),
                )
            ),
        }
    )


def _observation(offset: float = 0.0) -> dict[str, object]:
    return {
        "continuous": np.asarray([offset, -offset], dtype=np.float32),
        "nested": (
            1,
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([1, 3], dtype=np.int64),
        ),
    }


def test_gymnasium_collator_preserves_nested_structure_and_dtypes() -> None:
    batch = GymnasiumObservationCollator(_space()).collate_observations(
        [_observation(), _observation(0.5)]
    )

    assert batch["continuous"].shape == (2, 2)
    assert batch["continuous"].dtype == torch.float32
    assert batch["nested"][0].dtype == torch.int64
    assert batch["nested"][1].shape == (2, 2)
    assert batch["nested"][2].shape == (2, 2)


def test_gymnasium_collator_batches_box_tensors_without_numpy_round_trip() -> None:
    space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    values = [torch.tensor([0.25, -0.5]), torch.tensor([0.5, -0.25])]

    batch = GymnasiumObservationCollator(space).collate_observations(values)

    assert torch.equal(batch, torch.stack(values))


def test_gymnasium_collator_rejects_invalid_box_and_nested_values() -> None:
    collator = GymnasiumObservationCollator(_space())
    invalid = _observation()
    invalid["continuous"] = np.asarray([np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        collator.collate_observations([invalid])

    invalid = _observation()
    invalid["nested"] = (4, np.asarray([0, 1]), np.asarray([1, 3]))
    with pytest.raises(ValueError, match="Discrete"):
        collator.collate_observations([invalid])


def test_gymnasium_collator_rejects_invalid_shape_and_dtype() -> None:
    collator = GymnasiumObservationCollator(_space())
    invalid_shape = _observation()
    invalid_shape["continuous"] = np.asarray([0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        collator.collate_observations([invalid_shape])

    invalid_dtype = _observation()
    invalid_dtype["continuous"] = np.asarray([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="dtype"):
        collator.collate_observations([invalid_dtype])


def test_transition_collation_builds_standard_batch_once() -> None:
    observation = _observation()
    transitions = [
        Transition(observation, 1, 2.0, observation, False, False),
        Transition(observation, 0, 3.0, observation, True, False),
    ]

    batch = GymnasiumObservationCollator(_space()).collate_transitions(transitions)

    assert batch["observations"]["continuous"].shape == (2, 2)
    assert torch.equal(batch["actions"], torch.tensor([1, 0]))
    assert torch.equal(batch["rewards"], torch.tensor([2.0, 3.0]))
