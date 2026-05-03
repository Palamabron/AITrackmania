# standard library imports
import random
import time
from typing import Any, cast

# third-party imports
import gymnasium
from gymnasium import spaces
from loguru import logger
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

from tmrl.custom.interfaces import TM2020InterfaceBoundary

NB_STEPS = 1000
ACT_COMPUTE_MIN = 0.0
ACT_COMPUTE_MAX = 0.05


def benchmark():
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceBoundary
    env_config["benchmark"] = True
    env_config["running_average_factor"] = 0.05
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {
        "img_hist_len": 1,
        "gamepad": False,
    }
    env = gymnasium.make("real-time-gym-v1", config=env_config)

    t_d = time.time()
    _, _ = env.reset()
    for _idx in range(NB_STEPS - 1):
        _ = action_space.sample()  # simulate action compute time
        time.sleep(random.uniform(ACT_COMPUTE_MIN, ACT_COMPUTE_MAX))
        # o, r, d, t, i = env.step(act)
        step_out = cast(tuple[Any, ...], env.step(None))
        _o, r, d, t, _i, _s_r = step_out
        if d or t:
            env.reset()
        logger.info(f"rew:{r}")
    t_f = time.time()

    elapsed_time = t_f - t_d
    bench = getattr(env, "benchmarks", None)
    logger.info(f"benchmark results: {bench() if callable(bench) else 'n/a'}")
    logger.info(f"elapsed time: {elapsed_time}")
    logger.info(f"time-step duration: {elapsed_time / NB_STEPS}")


if __name__ == "__main__":
    benchmark()
