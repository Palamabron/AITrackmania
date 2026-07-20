from my_trackmania_agent.components import StarterMlpLearner


def test_policy_is_constructible() -> None:
    assert StarterMlpLearner().policy() is not None
