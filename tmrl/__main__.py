"""TMRL entrypoint: server, trainer, rollout worker, and utilities.

Use tyro CLI: e.g. python -m tmrl --server, python -m tmrl --trainer --wandb
"""

import json
import time
from dataclasses import dataclass

import tyro
from loguru import logger

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker, Server, Trainer
from tmrl.tools.check_environment import (
    check_env_tm20_trackmap,
    check_env_tm20full,
    check_env_tm20lidar,
)
from tmrl.tools.record_reward import record_reward_dist
from tmrl.util import partial


@dataclass
class TmrlCli:
    """Command-line interface for TMRL.

    Exactly one of the mode flags (install, server, trainer, worker, etc.)
    should be used per run. Optional modifiers (e.g. --wandb, --config)
    apply to the chosen mode.
    """

    install: bool = False
    """Check TMRL installation (prints TmrlData path)."""

    server: bool = False
    """Run the central server (buffer + model distribution)."""

    trainer: bool = False
    """Run the trainer (learn from replay, push model to server)."""

    worker: bool = False
    """Run a rollout worker (collect experience, pull model)."""

    expert: bool = False
    """Run an expert rollout worker (no model updates)."""

    test: bool = False
    """Run inference only (no training, standalone episodes)."""

    benchmark: bool = False
    """Benchmark the environment (no training)."""

    record_reward: bool = False
    """Record a reward function in TrackMania 2020."""

    use_keyboard: bool = False
    """Use keyboard (instead of gamepad) when recording reward."""

    record_episode: bool = False
    """Record an episode into the replay buffer."""

    check_env: bool = False
    """Verify environment (Lidar/Full/TrackMap) works."""

    wandb: bool = False
    """Enable Weights & Biases logging (use with --trainer)."""

    config: str = "{}"
    """JSON dict of rtgym config overrides, e.g. -d '{\"time_step_duration\": 0.1}'."""


def main(cli: TmrlCli) -> None:
    """Dispatch to the selected mode (server, trainer, worker, or utility)."""
    if cli.server:
        Server()
        while True:
            time.sleep(1.0)
    elif cli.worker or cli.test or cli.benchmark or cli.expert:
        env_config = cfg_obj.CONFIG_DICT.copy()
        try:
            config_overrides = json.loads(cli.config) if cli.config else {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid --config JSON: {e}")
            raise
        for key, value in config_overrides.items():
            env_config[key] = value
        rollout_worker = RolloutWorker(
            env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config}),
            actor_module_cls=cfg_obj.POLICY,
            sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
            device="cuda" if cfg.CUDA_INFERENCE else "cpu",
            server_ip=cfg.SERVER_IP_FOR_WORKER,
            max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
            crc_debug=cfg.CRC_DEBUG,
            standalone=cli.test,
        )
        if cli.worker:
            rollout_worker.run()
        elif cli.expert:
            rollout_worker.run(expert=True)
        elif cli.benchmark:
            rollout_worker.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rollout_worker.run_episodes(10000)
    elif cli.trainer:
        trainer = Trainer(
            training_cls=cfg_obj.TRAINER,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            model_path=cfg.MODEL_PATH_TRAINER,
            checkpoint_path=cfg.CHECKPOINT_PATH,
            dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
            load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
            updater_fn=cfg_obj.UPDATER_FN,
        )
        logger.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        if cli.wandb:
            trainer.run_with_wandb(
                entity=cfg.WANDB_ENTITY,
                project=cfg.WANDB_PROJECT,
                run_id=cfg.WANDB_RUN_ID,
            )
        else:
            trainer.run()
    elif cli.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=cli.use_keyboard)
    elif cli.check_env:
        if cfg.PRAGMA_LIDAR:
            if cfg.PRAGMA_TRACKMAP:
                check_env_tm20_trackmap()
            else:
                check_env_tm20lidar()
        else:
            check_env_tm20full()
    elif cli.record_episode:
        from tmrl.tools.record_episode import record_episode

        record_episode()
    elif cli.install:
        logger.info(f"TMRL folder: {cfg.TMRL_FOLDER}")
    else:
        raise ValueError(
            "No mode selected."
            " Use one of: "
            " --install, "
            " --server, "
            " --trainer, "
            " --worker, "
            " --test, "
            " --benchmark, "
            " --expert, "
            " --record-reward, "
            " --check-environment."
        )


if __name__ == "__main__":
    cli_args = tyro.cli(TmrlCli)
    main(cli_args)
