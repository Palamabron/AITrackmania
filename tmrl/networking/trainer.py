"""TrainerInterface, Trainer, and the main training loop functions."""

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from os.path import exists
from typing import Any

from loguru import logger
from tlspyo import Endpoint

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.actor import ActorModule
from tmrl.networking.buffer import Buffer
from tmrl.networking.utils import log_environment_variables, print_with_timestamp
from tmrl.util import dump, load, partial_to_dict

_dump_pids: list[int] = []
_dump_thread: "threading.Thread | None" = None


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path.
    On Unix, uses fork() so the parent returns immediately. On Windows (no fork),
    saves in a background thread so the server does not block and cause socket timeouts.
    """

    def _do_dump():
        try:
            dump(run_instance, checkpoint_path)
        except Exception as e:
            from loguru import logger

            logger.error(f"Error saving checkpoint in background: {e}")

    if hasattr(os, "fork"):
        global _dump_pids
        active_pids = []
        for pid in _dump_pids:
            try:
                pid_ret, _ = os.waitpid(pid, os.WNOHANG)
                if pid_ret == 0:
                    active_pids.append(pid)
            except ChildProcessError:
                pass
        _dump_pids = active_pids

        pid = os.fork()
        if pid == 0:
            try:
                dump(run_instance, checkpoint_path)
            except Exception as e:
                from loguru import logger

                logger.error(f"Error saving checkpoint in child process: {e}")
            finally:
                os._exit(0)
        else:
            _dump_pids.append(pid)
            return
    else:
        global _dump_thread
        dump_thread_join_timeout = 300
        if _dump_thread is not None and _dump_thread.is_alive():
            _dump_thread.join(timeout=dump_thread_join_timeout)
        import collections
        import copy

        snapshot = copy.copy(run_instance)
        snapshot.memory = copy.copy(run_instance.memory)
        snapshot.memory.data = [
            collections.deque(d, maxlen=run_instance.memory.memory_size)
            if isinstance(d, collections.deque)
            else copy.copy(d)
            for d in run_instance.memory.data
        ]
        if hasattr(snapshot.memory, "priorities"):
            snapshot.memory.priorities = copy.copy(run_instance.memory.priorities)
        if hasattr(snapshot.memory, "end_episodes_indices"):
            snapshot.memory.end_episodes_indices = copy.copy(
                run_instance.memory.end_episodes_indices
            )

        def _do_dump_safe():
            try:
                dump(snapshot, checkpoint_path)
            except Exception as e:
                from loguru import logger

                logger.error(f"Error saving checkpoint in background: {e}")

        _dump_thread = threading.Thread(target=_do_dump_safe, daemon=True)
        _dump_thread.start()


def iterate_epochs(
    run_cls,
    interface: "TrainerInterface",
    checkpoint_path: str | None,
    dump_run_instance_fn=dump_run_instance,
    load_run_instance_fn=load_run_instance,
    epochs_between_checkpoints=1,
    updater_fn=None,
):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logger.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logger.info("=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            from tmrl.config.run_artifacts import write_run_repro_bundle

            write_run_repro_bundle(checkpoint_path)
            logger.info("")
        else:
            logger.info("Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logger.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logger.info("Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logger.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            yield run_instance.run_epoch(interface=interface)

            if run_instance.epoch % epochs_between_checkpoints == 0:
                logger.info(" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logger.info(f" saved checkpoint in {time.time() - t1} seconds.")

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(
    entity,
    project,
    run_id,
    interface,
    run_cls,
    checkpoint_path: str | None = None,
    dump_run_instance_fn=None,
    load_run_instance_fn=None,
    updater_fn=None,
):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()
    import atexit

    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)
    import wandb

    cfg.ensure_wandb_api_key()
    logger.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config["environ"] = log_environment_variables()
    config["merged_config"] = cfg.merged_config_snapshot_redacted()
    hyperparams_dict = cfg.create_config()
    for key, value in hyperparams_dict.items():
        config[key] = value
    resume = bool(checkpoint_path and exists(checkpoint_path))
    wandb_initialized = False
    wandb_max_retries = 10
    wandb_retry_sleep = 10.0
    error_count = 0
    while not wandb_initialized:
        try:
            wandb.init(
                dir=wandb_dir,
                entity=entity,
                project=project,
                id=run_id + " TRAINER",
                resume=resume,
                config=config,
                job_type="trainer",
            )
            wandb.config.update(
                {"tmrl_validated_main_config": cfg.main_config_snapshot_redacted()},
                allow_val_change=True,
            )
            exp_id = os.environ.get("TMRL_EXPERIMENT_ID")
            if exp_id:
                wandb.config.update(
                    {"experiment_id": exp_id, "experiment_framework_version": "1.0"},
                    allow_val_change=True,
                )
                if wandb.run is not None:
                    wandb.run.tags = [
                        *list(wandb.run.tags or []),
                        f"exp:{exp_id}",
                    ]
            wandb_initialized = True

        except Exception as e:
            error_count += 1
            logger.warning(f"wandb error {error_count}: {e}")
            if error_count > wandb_max_retries:
                logger.warning("Could not connect to wandb, aborting.")
                import sys

                sys.exit(1)
            else:
                time.sleep(wandb_retry_sleep)
    for _stats in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        # Round-level stats are logged to wandb inside TrainingOffline.run_epoch() with
        # step=total_updates so steps stay monotonically increasing (agent logs per-batch).
        list(_stats)  # consume to drive the epoch


def run(
    interface,
    run_cls,
    checkpoint_path: str | None = None,
    dump_run_instance_fn=None,
    load_run_instance_fn=None,
    updater_fn=None,
):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for _ in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        pass


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """

    def __init__(
        self,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_com_port=cfg.LOCAL_PORT_TRAINER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
        model_path=cfg.MODEL_PATH_TRAINER,
    ):

        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else "127.0.0.1"
        self.__endpoint = Endpoint(
            ip_server=self.server_ip,
            port=server_port,
            password=password,
            groups="trainers",
            local_com_port=local_com_port,
            header_size=header_size,
            max_buf_len=max_buf_len,
            security=security,
            keys_dir=keys_dir,
            hostname=hostname,
            # Match worker: large Buffer pickles must deserialize on the training thread.
            # Async mode can silently drop messages on Windows when unpickling fails.
            deserializer_mode="synchronous",
        )

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={"trainers": -1})

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        if hasattr(model, "save_to_bytes"):
            weights = model.save_to_bytes()
        else:
            model.save(self.model_path)
            with open(self.model_path, "rb") as f:
                weights = f.read()
        self.__endpoint.broadcast(weights, "workers")

    def retrieve_buffer(self):
        """
        returns the TrainerInterface's buffer of training samples
        """
        buffers = self.__endpoint.receive_all()
        res = Buffer()
        for buf in buffers:
            res += buf
        self.__endpoint.notify(groups={"trainers": -1})
        if len(res) > 0:
            logger.debug("retrieve_buffer: got {} samples from server", len(res))
        return res


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """

    def __init__(
        self,
        training_cls=cfg_obj.TRAINER,
        server_ip=cfg.SERVER_IP_FOR_TRAINER,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_com_port=cfg.LOCAL_PORT_TRAINER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
        model_path=cfg.MODEL_PATH_TRAINER,
        checkpoint_path=cfg.CHECKPOINT_PATH,
        dump_run_instance_fn: Callable[..., Any] | None = None,
        load_run_instance_fn: Callable[..., Any] | None = None,
        updater_fn: Callable[..., Any] | None = None,
    ):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            server_port (int): public port of the central `Server`
            password (str): password of the central `Server`
            local_com_port (int): port used by `tlspyo` for local communication
            header_size (int): number of bytes used for `tlspyo` headers
            max_buf_len (int): maximum number of messages queued by `tlspyo`
            security (str): `tlspyo security type` (None or "TLS")
            keys_dir (str): custom credentials directory for `tlspyo` TLS security
            hostname (str): custom TLS hostname
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path for `Trainer` checkpoint (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, must be a
                function (checkpoint, training_cls) -> updated checkpoint, called after load.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(
            server_ip=server_ip,
            server_port=server_port,
            password=password,
            local_com_port=local_com_port,
            header_size=header_size,
            max_buf_len=max_buf_len,
            security=security,
            keys_dir=keys_dir,
            hostname=hostname,
            model_path=model_path,
        )

    def run(self):
        """
        Runs training.
        """
        run(
            interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn,
        )

    def run_with_wandb(
        self, entity=cfg.WANDB_ENTITY, project=cfg.WANDB_PROJECT, run_id=cfg.WANDB_RUN_ID, key=None
    ):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ["WANDB_API_KEY"] = key
        run_with_wandb(
            entity=entity,
            project=project,
            run_id=run_id,
            interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn,
        )
