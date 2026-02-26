# standard library imports
import atexit
import datetime
import itertools
import json
import os
import pickle
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from os.path import exists
from typing import Any

# third-party imports
import numpy as np
from loguru import logger
from requests import get  # type: ignore[import-untyped]
from tlspyo import Endpoint, Relay
from tlspyo.server import Server as TlspyoServer

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

# local imports
from tmrl.actor import ActorModule
from tmrl.util import dump, load, partial_to_dict

__docformat__ = "google"


def print_with_timestamp(message: str) -> None:
    """Log message with current date/time prefix."""
    timestamp = datetime.datetime.now().strftime("%x %X ")
    logger.info("{}{}", timestamp, message)


def print_ip():
    public_ip = get("http://api.ipify.org").text
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


def _start_relay_windows_tcp(
    port: int,
    password: str,
    local_port: int,
    header_size: int,
    max_workers: int,
):
    """Run tlspyo relay server in a thread on Windows when TLS is disabled.

    The default tlspyo Relay uses a subprocess for the server; on Windows the child's
    stderr is often not visible, so bind failures (e.g. port in use) are silent. This
    runs the same server logic in a thread so any exception is visible in the same process.
    """
    local_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_srv.bind(("127.0.0.1", local_port))
    local_srv.listen()

    accepted_groups = {
        "trainers": {"max_count": 1, "max_consumables": None},
        "workers": {"max_count": max_workers, "max_consumables": None},
    }
    server = TlspyoServer(
        port=port,
        password=password,
        serializer=pickle.dumps,
        deserializer=pickle.loads,
        accepted_groups=accepted_groups,
        local_com_port=local_port,
        header_size=header_size,
        security="TCP",
        keys_dir=None,
    )

    def run_server():
        # Twisted's reactor.run() installs signal handlers by default; that only works in the
        # main thread (ValueError on Windows/CPython). Patch the global reactor so the relay
        # thread runs with installSignalHandlers=0.
        from twisted.internet import reactor

        _orig_run = reactor.run

        def run_without_signals(install_signal_handlers=0):
            return _orig_run(installSignalHandlers=install_signal_handlers)

        reactor.run = run_without_signals
        try:
            server.run()
        except Exception as e:
            logger.exception(
                "Relay server thread failed (this is the process that should bind to port {}): {}",
                port,
                e,
            )
            raise

    thread = threading.Thread(target=run_server, daemon=False)
    thread.start()

    conn, _ = local_srv.accept()
    msg = server.serializer(("TEST", None))
    header = bytes(f"{len(msg):<{header_size}}", "utf-8")
    conn.sendall(header + msg)

    # Keep references so they are not GC'd; thread and conn must stay alive
    return type("_WindowsTcpRelay", (), {"_thread": thread, "_conn": conn, "_srv": local_srv})()


# BUFFER: ===========================================


class Buffer:
    """In-memory buffer of transition samples for the Server, RolloutWorker, and Trainer.

    Samples are tuples: (act, new_obs, rew, terminated, truncated, info).
    """

    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """Initialize an empty buffer with optional max length.

        Args:
            maxlen: Maximum number of samples to keep; older samples are dropped when exceeded.
        """
        self.memory = []
        self.stat_train_return = 0.0
        self.stat_test_return = 0.0
        self.stat_train_steps = 0
        self.stat_test_steps = 0
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen) :]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): (act, new_obs, rew, terminated, truncated, info)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# SERVER SERVER: =====================================


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """

    def __init__(
        self,
        port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_SERVER,
        header_size=cfg.HEADER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        max_workers=cfg.NB_WORKERS,
    ):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            max_workers (int): max number of accepted workers
        """
        if sys.platform == "win32" and security is None:
            # On Windows with TCP, run relay server in a thread so bind errors are visible
            # (tlspyo's Process-based relay often hides child stderr on Windows).
            self.__relay = _start_relay_windows_tcp(
                port=port,
                password=password,
                local_port=local_port,
                header_size=header_size,
                max_workers=max_workers,
            )
        else:
            self.__relay = Relay(
                port=port,
                password=password,
                accepted_groups={
                    "trainers": {"max_count": 1, "max_consumables": None},
                    "workers": {"max_count": max_workers, "max_consumables": None},
                },
                local_com_port=local_port,
                header_size=header_size,
                security=security,
                keys_dir=keys_dir,
            )
        # So that "Connected." and "New client with groups" from tlspyo appear in the server console
        import logging

        logging.getLogger("tlspyo").setLevel(logging.INFO)
        print_with_timestamp(
            f"TMRL server listening on port {port} (trainers + workers). "
            "Leave this process running."
        )
        try:
            config_path = str(cfg.CONFIG_FILE_PATH)
        except AttributeError:
            config_path = "(config path not available)"
        print_with_timestamp(
            f"Config: {config_path} (ensure server, trainer, worker use this same config)."
        )
        # Verify the relay subprocess actually bound to the port (e.g. no port conflict)
        time.sleep(0.5)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect(("127.0.0.1", port))
            print_with_timestamp(f"Port {port} is open and accepting connections.")
        except OSError as e:
            print_with_timestamp(
                f"WARNING: Could not connect to 127.0.0.1:{port}. "
                f"Relay subprocess may have failed to bind (port in use or firewall). Error: {e}"
            )

    def stop(self):
        """Stop the server so the process can exit (e.g. after Ctrl+C)."""
        relay = getattr(self, "_Server__relay", None)
        if relay is None:
            return
        if hasattr(relay, "_thread"):
            # Windows TCP in-thread relay: stop Twisted reactor and wait for thread
            try:
                from twisted.internet import reactor

                reactor.callFromThread(reactor.stop)
            except Exception:
                pass
            relay._thread.join(timeout=5.0)
        # If using tlspyo Relay subprocess, we don't hold a handle here; process exit will kill it


# TRAINER: ==========================================


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
        )

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={"trainers": -1})  # retrieve everything

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
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
        self.__endpoint.notify(groups={"trainers": -1})  # retrieve everything
        if len(res) > 0:
            logger.debug("retrieve_buffer: got {} samples from server", len(res))
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, "") for k in os.environ.get("LOG_VARIABLES", "").strip().split()}


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
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs(
    run_cls,
    interface: TrainerInterface,
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
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame

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
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(
        shutil.rmtree, wandb_dir, ignore_errors=True
    )  # clean up after wandb atexit handler finishes
    import wandb

    logger.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config["environ"] = log_environment_variables()
    hiperparams_dict = cfg.create_config()
    for key, value in hiperparams_dict.items():
        config[key] = value
    # config['git'] = git_info()  # TODO: check this for bugs
    resume = bool(checkpoint_path and exists(checkpoint_path))
    wandb_initialized = False
    err_cpt = 0
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
            wandb_initialized = True

        except Exception as e:
            err_cpt += 1
            logger.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logger.warning("Could not connect to wandb, aborting.")
                exit()
            else:
                time.sleep(10.0)
    # logger.info(config)
    global_step = 0
    for stats in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        for s in stats:
            log_dict = json.loads(s.to_json())
            # Ensure wandb receives serializable values (no NaN/Inf). Use 0.0 for
            # numeric metrics so curves are always plotted (WandB skips None).
            for k, v in list(log_dict.items()):
                is_invalid = (
                    v is None
                    or (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")))
                )
                if is_invalid:
                    log_dict[k] = 0.0 if (k.startswith("losses/") or k in (
                        "return_test", "return_train", "episode_length_test", "episode_length_train"
                    )) else None
            # Ensure key metrics exist (e.g. first round can have no batches → no keys)
            for key in (
                "losses/loss_actor", "losses/loss_critic",
                "return_test", "return_train", "episode_length_test", "episode_length_train",
            ):
                if key not in log_dict or log_dict[key] is None:
                    log_dict[key] = 0.0
            wandb.log(log_dict, step=global_step)
            global_step += 1


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
    for stats in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        pass


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


# ROLLOUT WORKER: ===================================


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """

    def __init__(
        self,
        env_cls,
        actor_module_cls,
        sample_compressor: Callable[..., Any] | None = None,
        device="cpu",
        max_samples_per_episode=np.inf,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor: Callable[..., Any] | None = None,
        crc_debug=False,
        model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
        model_history=cfg.MODEL_HISTORY,
        standalone=False,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_WORKER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): module class for the policy (tmrl.actor.ActorModule subclass)
            sample_compressor (callable): compressor for samples over the Internet; when not `None`,
                must take (prev_act, obs, rew, terminated, truncated, info) and return same order;
                works with a decompression scheme in the Memory class.
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): if not None, (obs) -> modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): path to policy history (omit .tmod); leave default recommended
            model_history (int): save policy every this many new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to(
            self.device
        )
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logger.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logger.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.start_time = time.time()
        self.server_ip = server_ip if server_ip is not None else "127.0.0.1"

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(
                ip_server=self.server_ip,
                port=server_port,
                password=password,
                groups="workers",
                local_com_port=local_port,
                header_size=header_size,
                max_buf_len=max_buf_len,
                security=security,
                keys_dir=keys_dir,
                hostname=hostname,
                deserializer_mode="synchronous",
            )
        else:
            self.__endpoint = None

    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        action = self.actor.act_(obs, test=test)
        # action = self.actor.act_(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        obs = None
        try:
            # Faster than hasattr() in real-time environments
            act = self.env.unwrapped.default_action  # .astype(np.float32)
        except AttributeError:
            # In non-real-time environments, act is None on reset
            act = None
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is obs -> act -> (new_obs, rew, terminated, truncated, info).
        In Real-Time RL, act is appended to a buffer that is part of new_obs (real-time delays).

        Args:
            reward_function:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)

        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(
                sample
            )  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects up to `max_samples` training transitions (reset to terminated or truncated).

        Stores the episode and training return in the worker's local Buffer.
        for sending to the `Server`.

        Args:
            max_samples (int): if not terminated after this many steps, reset and set truncated.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1
            )
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def run_episode(self, max_samples=None, train=False):
        """
        Collects up to `max_samples` test transitions (reset to terminated or truncated).

        Args:
            max_samples (int): at most this many samples per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=not train, collect_samples=False
            )
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        Sends episodes to the Server and checks for new weights between episodes.
        For sync/fine-grained sampling use other APIs; for deployment use run_episodes.

        Args:
            test_episode_interval (int): test episode every N train episodes; 0 to disable.
            nb_episodes (int): max train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): if True, send samples only, no model updates nor test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self.run_episode(self.max_samples_per_episode, train=False)
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            for episode in iterator:
                if (
                    test_episode_interval
                    and episode % test_episode_interval == 0
                    and not self.crc_debug
                ):
                    print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(
        self,
        test_episode_interval=0,
        nb_steps=np.inf,
        initial_steps=1,
        max_steps_per_update=np.inf,
        end_episodes=True,
        verbose=False,
    ):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        For traditional (non-real-time) envs that can be stepped fast.
        For rtgym with wait_on_done, set end_episodes to True.

        Note: Does not collect test episodes; use run_episode(train=False) periodically.

        Args:
            test_episode_interval (int): test every N train episodes; 0 to disable.
                Requires end_episodes.
            nb_steps (int): total steps to collect (after initial_steps).
            initial_steps (int): steps before waiting for first model update.
            max_steps_per_update (float): max steps per model from Server (can be non-integer).
            end_episodes (bool): if True, wait for episode end before send/wait; else pause.
            verbose (bool): whether to log INFO messages.
        """

        # collect initial samples

        if verbose:
            logger.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            # reset
            obs, info = self.reset(collect_samples=True)
            done = False
            iteration += 1
            # episode
            while not done and (end_episodes or iteration < initial_steps):
                # step
                obs, rew, terminated, truncated, info = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            # send the collected samples to the Server
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logger.info("Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        # wait for the first updated model if required here
        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logger.info(
                    f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates"
                )
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        # collect further samples while synchronizing with the Trainer

        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:
            if done:
                # test episode
                if (
                    test_episode_interval > 0
                    and episode % test_episode_interval == 0
                    and end_episodes
                ):
                    if verbose:
                        print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                # reset
                obs, info = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):
                # step
                obs, rew, terminated, truncated, info = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    # check model and send samples after each step
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logger.info(
                                f"Ratio {ratio} > {max_steps_per_update}, sending buffer (no eoe)"
                            )
                        if not done:
                            if verbose:
                                logger.info("Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                # check model and send samples only after episodes end
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logger.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer (eoe)")
                    if not done:
                        if verbose:
                            logger.info("Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logger.info(
                    f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}"
                )
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        The rtgym config must have the "benchmark" option set to True.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=test, collect_samples=False
            )
            if terminated or truncated:
                break
        res = self.env.unwrapped.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            with open(self.model_path, "wb") as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(
                        self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod",
                        "wb",
                    ) as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received
