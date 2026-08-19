# OpenPlanet integration

Copy `TrackmaniaRL_GrabData_IQN.as` to `OpenplanetNext/Scripts`, reload OpenPlanet,
and manually load the configured local map before running smoke tests or benchmarks.

Copy `.env-example` to `.env` and set a random `TRACKMANIARL_DISTRIBUTED_TOKEN`
with at least 32 characters before starting distributed processes.
