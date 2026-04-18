import cv2
import numpy as np
from gymnasium import spaces

from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.utils.control_mouse import mouse_save_replay_tm20
from tmrl.registry import INTERFACES


@INTERFACES.register("impala")
class TM2020InterfaceIMPALA(TM2020Interface):
    def __init__(
        self,
        img_hist_len=1,
        gamepad=False,
        record=False,
        save_replay: bool = False,
        grayscale: bool = False,
        resize_to: tuple = (128, 64),
        finish_reward: float = 0.0,
        constant_penalty: float = 0.05,
        crash_penalty: float = 0.0,
        checkpoint_reward: float = 0.0,
        lap_reward: float = 0.0,
        points_number: int = 5,
        **kwargs,
    ):
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            save_replays=save_replay,
            grayscale=grayscale,
            finish_reward=finish_reward,
            resize_to=resize_to,
            constant_penalty=constant_penalty,
            crash_penalty=crash_penalty,
            **kwargs,
        )
        self.record = record
        self.cur_lap = 0
        self.cur_checkpoint = 0
        self.lap_reward = lap_reward
        self.checkpoint_reward = checkpoint_reward
        self.points_number = points_number

    def get_observation_space(self):
        # https://gymnasium.farama.org/api/spaces/
        """Returns the observation space.

        Returns:
            observation_space: gymnasium.spaces.Tuple

        Note: Do NOT put the action buffer here (automated).
        Ensure rtgym config has act_buf_len and reset_act_buf set appropriately for RT-MDP.
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))

        input_steer = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_gas_pedal = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))

        acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))

        aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
        aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        race_progress = spaces.Box(low=0.0, high=1_000_000, shape=(1,))

        steer_angle = spaces.Box(low=-1000.0, high=1000.0, shape=(2,))  # fl, fr

        slip_coef = spaces.Box(low=0.0, high=1.0, shape=(2,))  # fl, fr

        gear = spaces.Box(low=0.0, high=6.0, shape=(1,))

        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))

        next_checkpoints = spaces.Box(low=-100.0, high=100.0, shape=(2 * self.points_number,))

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = self.window_width, self.window_height
        if self.grayscale:
            img = spaces.Box(
                low=0.0, high=255.0, shape=(self.img_hist_len, h, w)
            )  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(
                low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3)
            )  # cv2 images are (h, w, c)

        return spaces.Tuple(
            (
                next_checkpoints,
                speed,
                acceleration,
                jerk,
                race_progress,
                input_steer,
                input_gas_pedal,
                input_brake,
                gear,
                aim_yaw,
                aim_pitch,
                steer_angle,
                slip_coef,
                failure_counter,
                img,
            )
        )

    def grab_data_and_img(self, percentage_to_cut: float = 0.2):
        assert self.window_interface is not None
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        height, _ = img.shape[:2]
        cut_height = int(height * percentage_to_cut)
        img = img[cut_height:, :]
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img[:, :, ::-1]
        )  # reversed view for numpy RGB convention
        data = self.grab_data()
        # print(f"data: {data}")
        self.img = img  # for render()
        # cv2.imshow("Environment", img)
        # cv2.waitKey(1)
        return data, img

    def grab_data(self):
        assert self.client is not None
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        assert self.reward_function is not None and self.img_hist is not None
        # print(f"data: {data}")
        cur_cp = int(data[0])
        cur_lap = int(data[1])

        speed = np.array([data[2]], dtype="float32")

        pos = np.array([data[3], data[4], data[5]], dtype="float32")

        input_steer = np.array([data[6]], dtype="float32")
        input_gas_pedal = np.array([data[7]], dtype="float32")
        input_brake = np.array([data[8]], dtype="float32")

        acceleration = np.array([data[10]], dtype="float32")
        jerk = np.array([data[11]], dtype="float32")

        aim_yaw = np.array([data[12]], dtype="float32")
        aim_pitch = np.array([data[13]], dtype="float32")

        steer_angle = np.array(data[14:16], dtype="float32")

        slip_coef = np.array(data[16:18], dtype="float32")

        gear = np.array([data[18]], dtype="float32")

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=pos,  # position x,y,z
            crashed=bool(self.is_crashed),
            speed=speed[0],
            next_cp=self.cur_checkpoint < cur_cp,
            next_lap=self.cur_lap < cur_lap,
        )
        failure_counter_val = float(failure_counter)

        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        race_progress_scalar = self.reward_function.compute_race_progress()

        next_checkpoints = self.reward_function.get_n_next_checkpoints_xy(pos, self.points_number)

        end_of_track = bool(data[9])

        if not self.is_crashed:
            self.crash_cooldown -= 1

        if end_of_track:
            terminated = True
            failure_counter_val = 0.0
            if self.save_replays:
                mouse_save_replay_tm20(True)

        race_progress_arr = np.array([race_progress_scalar], dtype="float32")

        failure_counter_arr = np.array([failure_counter_val], dtype=np.float32)
        info = {"reward_sum": reward_sum, "end_of_track": bool(end_of_track)}
        if getattr(self.client, "_last_retrieve_invalid", False):
            terminated = True
            info["telemetry_invalid"] = True
        if getattr(self.client, "_last_retrieve_position_patched", False):
            info["position_patched"] = True

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress_arr,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter_arr,
            imgs,
        ]

        total_obs = [next_checkpoints, *observation]

        total_obs[0] = np.array(total_obs[0])

        reward = np.float32(float(rew))
        # print(f"Reward: {reward}, crashed {bool(crashed)}, race progress {...}")
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        # if options['pad'] is not None and options['pad']:
        #     self.gamepad = None
        #     if not self.initialized:
        #         self.initialize()
        #     print(f"Restart the map")
        #     time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1)
        #     time.sleep(time_sleep)
        # else:
        self.reset_common()
        data, img = self.grab_data_and_img()
        assert self.reward_function is not None and self.img_hist is not None

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed = np.array([data[2]], dtype="float32")

        pos = np.array([data[3], data[4], data[5]], dtype="float32")

        input_steer = np.array([data[6]], dtype="float32")
        input_gas_pedal = np.array([data[7]], dtype="float32")
        input_brake = np.array([data[8]], dtype="float32")
        # isFinished 9
        acceleration = np.array([data[10]], dtype="float32")
        jerk = np.array([data[11]], dtype="float32")

        aim_yaw = np.array([data[12]], dtype="float32")
        aim_pitch = np.array([data[13]], dtype="float32")

        steer_angle = np.array(data[14:16], dtype="float32")

        slip_coef = np.array(data[16:18], dtype="float32")

        gear = np.array([data[18]], dtype="float32")

        failure_counter = np.array([0.0])
        race_progress = np.array([0.0], dtype="float32")

        next_checkpoints = self.reward_function.get_n_next_checkpoints_xy(pos, self.points_number)

        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter,
            imgs,
        ]

        total_obs = [next_checkpoints, *observation]

        total_obs[0] = np.array(total_obs[0])

        self.reward_function.reset()
        info = {"reward_sum": 0.0}
        return total_obs, info
