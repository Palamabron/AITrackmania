import platform
import time

if platform.system() == "Windows":
    from pyautogui import click, mouseUp

    def mouse_close_finish_pop_up_tm20(small_window: bool = False):
        """Click the finish pop-up dismiss button in TM2020.

        Args:
            small_window: If True, use small-window pixel coordinates;
                otherwise use full-size window coordinates.
        """
        if small_window:
            click(138, 108)
        else:
            click(550, 320)  # clicks where the "improve" button is supposed to be
        mouseUp()

    def mouse_change_name_replay_tm20(small_window: bool = False):
        """Double-click the replay name field to select it for renaming.

        Args:
            small_window: If True, use small-window pixel coordinates;
                otherwise use full-size window coordinates.
        """
        if small_window:
            click(138, 124)
            click(138, 124)
        else:
            click(500, 390)
            click(500, 390)

    def mouse_save_replay_tm20(small_window: bool = False):
        """Click through the TM2020 replay-save confirmation dialog.

        Waits 5 s for the save dialog to appear, clicks the save button,
        waits 0.2 s, then clicks the confirmation button.

        Args:
            small_window: If True, use small-window pixel coordinates;
                otherwise use full-size window coordinates.
        """
        time.sleep(5.0)
        if small_window:
            click(130, 110)
            mouseUp()
            time.sleep(0.2)
            click(130, 108)
            mouseUp()
        else:
            click(500, 345)
            mouseUp()
            time.sleep(0.2)
            click(500, 320)
            mouseUp()

    def mouse_close_replay_window_tm20(small_window: bool = False):
        """Click the close button on the TM2020 replay window.

        Args:
            small_window: If True, use small-window pixel coordinates;
                otherwise use full-size window coordinates.
        """
        if small_window:
            click(130, 95)
        else:
            click(500, 280)
        mouseUp()

else:

    def mouse_close_finish_pop_up_tm20(small_window: bool = False):
        pass

    def mouse_change_name_replay_tm20(small_window: bool = False):
        pass

    def mouse_save_replay_tm20(small_window: bool = False):
        pass

    def mouse_close_replay_window_tm20(small_window: bool = False):
        pass


if __name__ == "__main__":
    import time

    mouse_save_replay_tm20()
