"""Interface for TQC_GrabData OpenPlanet plugin (20 floats: API state + isCrashed + gear)."""

import numpy as np

import tmrl.config.config_constants as cfg
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

# TQC_GrabData plugin sends 20 floats per frame:
# 0: curCP, 1: curLap, 2: speed, 3-5: pos xyz, 6: steer, 7: gas, 8: brake,
# 9: isFinished, 10: acceleration, 11: jerk, 12: aimYaw, 13: aimPitch,
# 14-15: FL/FR steer angle, 16-17: FL/FR slip, 18: isCrashed, 19: gear
TQC_GRAB_NB_FLOATS = 20

# Fallback if REWARD_CONFIG["MIN_STEPS"] is missing (min steps before we trust end_of_track).
_DEFAULT_MIN_STEPS_END_OF_TRACK = 50
# Hard guarantee: never end episode before this many steps (avoids any source of early reset).
_MIN_EPISODE_LENGTH_GUARANTEED = 100


class TM2020InterfaceTQC(TM2020InterfaceIMPALASophy):
    """Uses TQC_GrabData plugin: port 9000, 20 floats; isCrashed at index 18, gear at 19.

    Keep the TrackMania window focused during training. If it loses focus the game may
    pause, rtgym will report time-step timeouts, and the worker will not reset until
    the game is focused again.
    """

    def initialize_common(self):
        # Create our client first so base does not create a second (base only if client is None).
        # Otherwise two client threads; plugin accepts one and we might read from the other.
        self.client = TM2020OpenPlanetClient(port=9000, nb_floats=TQC_GRAB_NB_FLOATS)
        super().initialize_common()

    def get_obs_rew_terminated_info(self):
        data = self.grab_data()
        # TQC_GrabData: index 18 = isCrashed, 19 = gear
        self.is_crashed = bool(data[18])
        cur_cp = int(data[0])
        cur_lap = int(data[1])

        speed = np.array([data[2] * 3.6], dtype="float32")
        pos = np.array([data[3], data[4], data[5]], dtype="float32")
        input_steer = np.array([data[6]], dtype="float32")
        input_gas_pedal = np.array([data[7]], dtype="float32")
        input_brake = np.array([data[8]], dtype="float32")
        end_of_track = bool(data[9])
        acceleration = np.array([data[10]], dtype="float32")
        jerk = np.array([data[11]], dtype="float32")
        aim_yaw = np.array([data[12]], dtype="float32")
        aim_pitch = np.array([data[13]], dtype="float32")
        steer_angle = np.array(data[14:16], dtype="float32")
        slip_coef = np.array(data[16:18], dtype="float32")
        gear = np.array([data[19]], dtype="float32")  # index 19 in TQC_GrabData

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=pos,
            crashed=self.is_crashed,
            speed=speed[0],
            next_cp=self.cur_checkpoint < cur_cp,
            next_lap=self.cur_lap < cur_lap,
            end_of_tack=end_of_track,
        )
        self._dbg_last_step = {
            "terminated": bool(terminated),
            "end_of_track": bool(end_of_track),
            "speed_kmh": float(speed[0]),
            "reward_sum": float(reward_sum),
            "step_counter": int(getattr(self.reward_function, "step_counter", -1)),
        }

        race_progress = self.reward_function.compute_race_progress()

        # Minimum run length: do not end on end_of_track for the first N steps (same N as
        # REWARD_CONFIG["MIN_STEPS"]). Gives the policy time to act and avoids instant
        # restarts when the model outputs no gas at the start.
        self._steps_since_reset = getattr(self, "_steps_since_reset", 0) + 1
        min_steps_before_finish = max(
            _DEFAULT_MIN_STEPS_END_OF_TRACK,
            cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if end_of_track and self._steps_since_reset >= min_steps_before_finish:
            terminated = True
            failure_counter = 0.0
            if self.save_replays:
                from tmrl.custom.utils.control_mouse import mouse_save_replay_tm20

                mouse_save_replay_tm20(True)

        self.reward_function.log_model_run(terminated=terminated, end_of_track=end_of_track)

        left_track, center_track, right_track = self.reward_function.get_track_info(
            pos, self.points_number
        )

        if not self.is_crashed:
            self.crash_cooldown -= 1

        race_progress = np.array([race_progress], dtype="float32")
        failure_counter = np.array([float(failure_counter)])
        info = {"reward_sum": reward_sum}

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
        ]
        track_info = [left_track + center_track + right_track]
        total_obs = track_info + observation
        total_obs[0] = np.array(total_obs[0])

        self.cur_checkpoint = cur_cp
        self.cur_lap = cur_lap

        # Hard guarantee: do not end episode in the first N steps (stops all early-reset loops).
        min_length = max(
            _MIN_EPISODE_LENGTH_GUARANTEED,
            2 * cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if self._steps_since_reset < min_length:
            terminated = False

        reward = np.float32(rew)
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        # If the environment truncates the episode (e.g. max length) without terminated=True,
        # log the run before resetting so "Total reward of the run" is not lost.
        if (
            getattr(self, "reward_function", None) is not None
            and getattr(self.reward_function, "step_counter", 0) > 0
            and not getattr(self.reward_function, "_logged_run_this_episode", False)
        ):
            self.reward_function.log_model_run(terminated=True, end_of_track=False)
        self.reset_common()
        self._steps_since_reset = 0
        data = self.grab_data()

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed = np.array([data[2] * 3.6], dtype="float32")
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
        gear = np.array([data[19]], dtype="float32")

        failure_counter = np.array([0.0])
        race_progress = np.array([0.0], dtype="float32")

        left_track, center_track, right_track = self.reward_function.get_track_info(
            pos, self.points_number
        )

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
        ]
        track_info = [left_track + center_track + right_track]
        total_obs = track_info + observation
        total_obs[0] = np.array(total_obs[0])

        self.reward_function.reset()
        info = {"reward_sum": 0.0}
        return total_obs, info
