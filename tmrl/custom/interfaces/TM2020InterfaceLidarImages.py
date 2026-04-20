"""LIDAR + progress + camera images for fusion models."""

from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidarConfigurable


class TM2020InterfaceLidarProgressImages(TM2020InterfaceLidarConfigurable):
    """
    LIDAR + progress + camera images from the same screenshot.
    Observation: (speed, progress, lidar_history, image_history).
    """

    def __init__(
        self,
        img_hist_len: int = 4,
        gamepad: bool = False,
        grayscale: bool = True,
        resize_to: tuple | None = None,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
        **kwargs,
    ):
        super().__init__(
            include_progress=True,
            include_camera_images=True,
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replays,
            grayscale=grayscale,
            resize_to=resize_to,
            **kwargs,
        )
