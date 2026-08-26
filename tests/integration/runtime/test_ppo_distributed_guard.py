from types import SimpleNamespace

import pytest

from trackmaniarl.distributed.coordinator import Coordinator


def test_distributed_coordinator_rejects_on_policy_learner() -> None:
    run = SimpleNamespace(learner=SimpleNamespace(on_policy=True))

    with pytest.raises(ValueError, match="does not support on-policy learners"):
        Coordinator(
            run,
            bind="127.0.0.1:8787",
            token="tests-only-distributed-token-0123456789",
            fingerprint="fingerprint",
        )
