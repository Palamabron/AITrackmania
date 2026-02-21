# third-party imports
# from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset
# third-party imports
import rtgym

# local imports
import tmrl.config.config_constants as cfg
import tmrl.custom.models.IMPALA as impala
import tmrl.custom.models.Sophy as impalaWoImages
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import TQCAgent as TQC_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.custom_memories import (
    MemoryTMLidar,
    MemoryTMLidarProgress,
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_progress,
    get_local_buffer_sample_tm20_imgs,
    MemoryTMBest,
    get_local_buffer_sample_mobilenet,
    MemoryTMFull,
    MemoryR2D2,
    MemoryR2D2woImages,
)
from tmrl.custom.tm.tm_preprocessors import (
    obs_preprocessor_tm_act_in_obs,
    obs_preprocessor_tm_lidar_act_in_obs,
    obs_preprocessor_tm_lidar_progress_act_in_obs,
    obs_preprocessor_mobilenet_act_in_obs,
)
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA
from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress
from tmrl.custom.interfaces.TM2020InterfaceTrackMap import TM2020InterfaceTrackMap
from tmrl.custom.custom_models import (
    MLPActorCritic,
    SquashedGaussianMLPActor,
    REDQMLPActorCritic,
    RNNActorCritic,
    SquashedGaussianRNNActor,
    VanillaCNNActorCritic,
    SquashedGaussianVanillaCNNActor,
    VanillaColorCNNActorCritic,
    SquashedGaussianVanillaColorCNNActor,
)
from tmrl.envs import GenericGymEnv
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]

MODEL_CONFIG = cfg.TMRL_CONFIG["MODEL"]

assert ALG_NAME in ["SAC", "REDQSAC",
                    "TQC"], f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."

# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic
        POLICY = SquashedGaussianMLPActor
else:
    if cfg.PRAGMA_MBEST_TQC:
        assert ALG_NAME in ("TQC", "SAC"), f"{ALG_NAME} is not implemented here."
        if cfg.USE_IMAGES:
            TRAIN_MODEL = impala.QRCNNActorCritic
            POLICY = impala.SquashedActorQRCNN
        else:
            TRAIN_MODEL = impalaWoImages.SophyActorCritic
            POLICY = impalaWoImages.SquashedActorSophy
    else:
        assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = SquashedGaussianVanillaCNNActor if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        INT = partial(TM2020InterfaceLidarProgress, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    elif cfg.PRAGMA_TRACKMAP:
        INT = partial(TM2020InterfaceTrackMap, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    else:
        INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        if cfg.USE_IMAGES:
            INT = partial(
                TM2020InterfaceIMPALA, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE, resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                crash_penalty=cfg.CRASH_PENALTY, constant_penalty=cfg.CONSTANT_PENALTY,
                checkpoint_reward=cfg.CHECKPOINT_REWARD, lap_reward=cfg.LAP_REWARD,
                min_nb_steps_before_failure=cfg.MIN_NB_STEPS_BEFORE_FAILURE
            )
        else:
            INT = partial(
                TM2020InterfaceIMPALASophy, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD,
                grayscale=cfg.GRAYSCALE, resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                crash_penalty=cfg.CRASH_PENALTY, constant_penalty=cfg.CONSTANT_PENALTY,
                checkpoint_reward=cfg.CHECKPOINT_REWARD, lap_reward=cfg.LAP_REWARD,
                min_nb_steps_before_failure=cfg.MIN_NB_STEPS_BEFORE_FAILURE
            )
    else:
        INT = partial(TM2020Interface,
                      img_hist_len=cfg.IMG_HIST_LEN,
                      gamepad=cfg.PRAGMA_GAMEPAD,
                      grayscale=cfg.GRAYSCALE,
                      resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# to compress a sample before sending it over the local network/Internet:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_mobilenet
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# to preprocess observations that come out of the gymnasium environment:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        OBS_PREPROCESSOR = obs_preprocessor_mobilenet_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer:
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        if cfg.PRAGMA_PROGRESS:
            MEM = MemoryTMLidarProgress
        else:
            MEM = MemoryTMLidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC:
        MEM = MemoryTMBest
    elif cfg.PRAGMA_MBEST_TQC:
        if cfg.USE_IMAGES:
            MEM = MemoryR2D2
        else:
            MEM = MemoryR2D2woImages
    else:
        MEM = MemoryTMFull

MEMORY = partial(MEM,
                 memory_size=MODEL_CONFIG["MEMORY_SIZE"],
                 batch_size=MODEL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG)

# ALGORITHM: ===================================================

if ALG_NAME == "SAC":
    AGENT = partial(
        SAC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None
    )
elif ALG_NAME == "TQC":
    AGENT = partial(
        TQC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        top_quantiles_to_drop=ALG_CONFIG["TOP_QUANTILES_TO_DROP"],
        quantiles_number=ALG_CONFIG["QUANTILES_NUMBER"],
        n_steps=ALG_CONFIG["N_STEPS"]
    )
else:
    AGENT = partial(
        REDQ_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        n=ALG_CONFIG["REDQ_N"],  # number of Q networks
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )


# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=MODEL_CONFIG["MAX_EPOCHS"],
        rounds=MODEL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=MODEL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=MODEL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=MODEL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=MODEL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        python_profiling=cfg.PROFILE_TRAINER,
        pytorch_profiling=cfg.PYTORCH_PROFILER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=MODEL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  # set this > 0 to start from an existing
    # policy (fills the buffer up to this number of samples before starting training)
else:  # images
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=MODEL_CONFIG["MAX_EPOCHS"],
        rounds=MODEL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=MODEL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=MODEL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=MODEL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=MODEL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        python_profiling=cfg.PROFILE_TRAINER,
        pytorch_profiling=cfg.PYTORCH_PROFILER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=MODEL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC", "TQC"] else None