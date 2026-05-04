import cv2
import gymnasium
from loguru import logger
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

import tmrl.config as cfg
from tmrl.custom.interfaces import TM2020Interface, TM2020InterfaceBoundary

_CHECK_ROUNDS = 1200


def check_env_tm20_boundary():
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceBoundary
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {
        "img_hist_len": 1,
        "gamepad": False,
        "record": False,
    }
    env = gymnasium.make("real-time-gym-v1", config=env_config)
    _, _ = env.reset()
    current = 0
    while current < _CHECK_ROUNDS:
        current += 1
        _o, r, d, t, _ = env.step(None)
        logger.info(f"r:{r}, d:{d}")

        if d or t:
            print("d: ", d)
            print("t: ", t)
            _o, _ = env.reset()


def show_imgs(imgs, scale=cfg.IMG_SCALE_CHECK_ENV):
    """Stack an image history vertically and display it at ``scale`` via OpenCV.

    Accepts either grayscale stacks ``(N, H, W)`` or colour stacks ``(N, H, W, C)``.
    """
    imshape = imgs.shape
    if len(imshape) == 3:
        nb, h, w = imshape
        concat = imgs.reshape((nb * h, w))
        width = int(concat.shape[1] * scale)
        height = int(concat.shape[0] * scale)
        cv2.imshow(
            "Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST)
        )
        cv2.waitKey(1)
    elif len(imshape) == 4:
        nb, h, w, c = imshape
        concat = imgs.reshape((nb * h, w, c))
        width = int(concat.shape[1] * scale)
        height = int(concat.shape[0] * scale)
        cv2.imshow(
            "Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST)
        )
        cv2.waitKey(1)


def check_env_tm20full():
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020Interface
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {
        "gamepad": False,
        "grayscale": cfg.GRAYSCALE,
        "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    }
    env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)
    o, _ = env.reset()
    show_imgs(o[3])
    logger.info(
        f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]"
    )
    while True:
        o, r, d, t, _ = env.step(None)
        show_imgs(o[3])
        logger.info(
            f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, "
            f"{o[2].item():07.01f}, imgs({len(o[3])})]"
        )
        if d or t:
            o, _ = env.reset()
            show_imgs(o[3])
            logger.info(
                f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, "
                f"imgs({len(o[3])})]"
            )


if __name__ == "__main__":
    check_env_tm20_boundary()
