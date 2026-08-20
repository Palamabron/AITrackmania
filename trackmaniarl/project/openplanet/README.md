# OpenPlanet integration

Copy `TrackmaniaRL_GrabData_IQN.as` to `OpenplanetNext/Scripts`, reload OpenPlanet,
and manually load the configured local map before running smoke tests or benchmarks.
The historical script filename is retained for installation compatibility; its
telemetry protocol is algorithm-neutral and serves RunSpec 2.0 Q/QR-DQN/IQN/FQF,
behavior-cloning and continuous-control models.

Copy `.env-example` to `.env` and set a random `TRACKMANIARL_DISTRIBUTED_TOKEN`
with at least 32 characters before starting distributed processes.
