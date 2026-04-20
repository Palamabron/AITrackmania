from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidarConfigurable


class TM2020InterfaceLidarProgress(TM2020InterfaceLidarConfigurable):
    def __init__(
        self,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
        **kwargs,
    ):
        super().__init__(
            include_progress=True,
            include_camera_images=False,
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replays,
            **kwargs,
        )
