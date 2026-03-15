"""Best TrackMania memory implementation with all telemetry fields."""

import numpy as np

from tmrl.custom.memories.base import MemoryTM, last_true_in_list, replace_hist_before_eoe


class MemoryTMBest(MemoryTM):
    """Comprehensive TrackMania memory with full telemetry data.

    This memory stores all available telemetry fields from the game,
    including detailed physics information for advanced algorithms.
    """

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # Handle reset transitions
        last_eoes = self.data[27][idx_now - self.min_samples : idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(
                hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1
            )
            replace_hist_before_eoe(
                hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset
            )
            replace_hist_before_eoe(
                hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1
            )
            replace_hist_before_eoe(
                hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset
            )

        last_obs = (
            self.data[2][idx_last],  # position
            self.data[3][idx_last],  # speed
            self.data[4][idx_last],  # acceleration
            self.data[5][idx_last],  # jerk
            self.data[6][idx_last],  # race_progress
            self.data[7][idx_last],  # input_steer
            self.data[8][idx_last],  # input_gas_pedal
            self.data[9][idx_last],  # input_brake
            self.data[10][idx_last],  # gear
            self.data[11][idx_last],  # aim_yaw
            self.data[12][idx_last],  # aim_pitch
            self.data[13][idx_last],  # surface_id
            self.data[14][idx_last],  # steer_angle
            self.data[15][idx_last],  # wheel_rot
            self.data[16][idx_last],  # wheel_rot_speed
            self.data[17][idx_last],  # damper_len
            self.data[18][idx_last],  # slip_coef
            self.data[19][idx_last],  # reactor_ground_mode
            self.data[20][idx_last],  # ground_contact
            self.data[21][idx_last],  # reactor_air_control
            self.data[22][idx_last],  # ground_dist
            self.data[23][idx_last],  # crashed
            self.data[24][idx_last],  # failure counter
            self.data[25][idx_last],  # imgs
            *last_act_buf,
        )
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[28][idx_now])
        new_obs = (
            self.data[2][idx_now],
            self.data[3][idx_now],
            self.data[4][idx_now],
            self.data[5][idx_now],
            self.data[6][idx_now],
            self.data[7][idx_now],
            self.data[8][idx_now],
            self.data[9][idx_now],
            self.data[10][idx_now],
            self.data[11][idx_now],
            self.data[12][idx_now],
            self.data[13][idx_now],
            self.data[14][idx_now],
            self.data[15][idx_now],
            self.data[16][idx_now],
            self.data[17][idx_now],
            self.data[18][idx_now],
            self.data[19][idx_now],
            self.data[20][idx_now],
            self.data[21][idx_now],
            self.data[22][idx_now],
            self.data[23][idx_now],
            self.data[24][idx_now],
            self.data[25][idx_now],
            *new_act_buf,
        )
        terminated = self.data[30][idx_now]
        truncated = self.data[31][idx_now]
        info = self.data[29][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence for a transition.

        Args:
            item: Starting index for loading images.

        Returns:
            Stacked array of images normalized to [0, 1].
        """
        res = self.data[26][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item: int):
        """Load action sequence for a transition.

        Args:
            item: Starting index for loading actions.

        Returns:
            Array of actions.
        """
        res = self.data[1][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Args:
            buffer: Buffer containing (act, obs, rew, terminated, truncated, info) samples.

        Returns:
            Self for method chaining.
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [np.array([b[1][0]]) for b in buffer.memory]  # position
        d3 = [np.array([b[1][1]]) for b in buffer.memory]  # speed
        d4 = [np.array([b[1][2]]) for b in buffer.memory]  # acceleration
        d5 = [np.array([b[1][3]]) for b in buffer.memory]  # jerk
        d6 = [np.array([b[1][4]]) for b in buffer.memory]  # race_progress
        d7 = [np.array([b[1][5]]) for b in buffer.memory]  # input_steer
        d8 = [np.array([b[1][6]]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array([b[1][7]]) for b in buffer.memory]  # input_brake
        d10 = [np.array([b[1][8]]) for b in buffer.memory]  # gear
        d11 = [np.array([b[1][9]]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array([b[1][10]]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array([b[1][12][0]]) for b in buffer.memory]  # surface_id
        d14 = [np.array([b[1][13][0]]) for b in buffer.memory]  # steer_angle
        d15 = [np.array([b[1][14][0]]) for b in buffer.memory]  # wheel_rot
        d16 = [np.array([b[1][15][0]]) for b in buffer.memory]  # wheel_rot_speed
        d17 = [np.array(b[1][16]) for b in buffer.memory]  # damper_len
        d18 = [np.array(b[1][17]) for b in buffer.memory]  # slip_coef
        d19 = [np.array([b[1][18]]) for b in buffer.memory]  # reactor_ground_mode
        d20 = [np.array([b[1][19]]) for b in buffer.memory]  # ground_contact
        d21 = [np.array(b[1][20]) for b in buffer.memory]  # reactor_air_control
        d22 = [np.array([b[1][21]]) for b in buffer.memory]  # ground_dist
        d23_list = [b[1][22].tolist() for b in buffer.memory]  # crashed
        d23 = [np.array([el]) for el in d23_list]
        d24 = [np.array([b[1][23]]) for b in buffer.memory]  # failure counter
        d25 = [b[1][24] for b in buffer.memory]  # imgs
        d26 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d27 = [b[2] for b in buffer.memory]  # rewards
        d28 = [b[5] for b in buffer.memory]  # infos
        d29 = [b[3] for b in buffer.memory]  # terminated
        d30 = [b[4] for b in buffer.memory]  # truncated

        data_fields = [
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
            d15,
            d16,
            d17,
            d18,
            d19,
            d20,
            d21,
            d22,
            d23,
            d24,
            d25,
            d26,
            d27,
            d28,
            d29,
            d30,
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(31):
                self.data[i] = self.data[i][to_trim:]

        return self
