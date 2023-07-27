import numpy as np
from gymnasium import spaces
from scipy import spatial

from custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from custom.utils.control_mouse import mouse_save_replay_tm20


class TM2020InterfaceTrackMap(TM2020InterfaceLidar):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5), record=False,
                 save_replay: bool = False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay)
        self.record = record
        self.window_interface = None
        self.lidar = None
        self.last_pos = [0, 0]
        self.index = 0
        self.map_left = np.loadtxt(r'D:/===Studia===/inzynier/tmrlBase/tmrl/saved_tracks/tmrl-test/track_left.csv',
                                   delimiter=',')
        self.map_right = np.loadtxt(r'D:/===Studia===/inzynier/tmrlBase/tmrl/saved_tracks/tmrl-test/track_right.csv',
                                    delimiter=',')
        self.all_observed_track_parts = [[], [], [], [], []]

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        track_information = spaces.Box(low=-300, high=300, shape=(60,))
        acceleration = spaces.Box(low=-100, high=100.0, shape=(1,))
        steering_angle = spaces.Box(low=-1, high=1.0, shape=(1,))
        slipping_tires = spaces.Box(low=0.0, high=1, shape=(4,))
        crash = spaces.Box(low=0.0, high=1, shape=(1,))
        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))
        return spaces.Tuple(
            (
                speed, gear, rpm, track_information, acceleration,
                steering_angle, slipping_tires, crash, failure_counter
            )
        )

    def grab_data(self):
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):

        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """

        data = self.grab_data()
        # print(f"data: {data}")
        car_position = [data[2], data[4]]
        yaw = data[11]  # angle the car is facing
        # if self.last_pos == car_position:
        #     print("package loss or something")
        self.last_pos = car_position
        # retrieving map information --------------------------------------
        # Cut out a portion directly in front of the car, as input for the ai
        look_ahead_distance = 15  # points to look ahead on the track
        nearby_correction = 60  # one point on a side needs to be at least this close to the same point on the other side
        l_x, l_z, r_x, r_z = self.get_track_in_front(car_position, look_ahead_distance, nearby_correction)

        # normalize the track in front

        l_x, l_z, r_x, r_z = self.normalize_track(l_x, l_z, r_x, r_z, car_position, yaw)
        # save the track in front in a file, so we can play it back later
        self.all_observed_track_parts[0].append(l_x.tolist())
        self.all_observed_track_parts[1].append(l_z.tolist())
        self.all_observed_track_parts[2].append(r_x.tolist())
        self.all_observed_track_parts[3].append(r_z.tolist())
        self.all_observed_track_parts[4].append(car_position)
        # ----------------------------------------------------------------------

        track_information = np.array(np.append(np.append(l_x, r_x), np.append(l_z, r_z)), dtype='float32')
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        acceleration = np.array([
            data[18],
        ], dtype='float32')
        steering_angle = np.array([
            data[19],
        ], dtype='float32')
        slipping_tires = np.array(data[20:24], dtype='float32')
        crash = np.array([
            data[24],
        ], dtype='float32')
        # self.isFirstTime = False
        end_of_track = bool(data[8])
        info = {}
        crash_penalty = -10  # < 0 to give a penalty
        reward = 0
        if crash == 1:
            reward += crash_penalty
        # print("crash penalty was not given")
        if end_of_track:
            reward += self.finish_reward
            terminated = True
            failure_counter = 0
            if self.save_replays:
                mouse_save_replay_tm20()
        else:
            rew, terminated, failure_counter = self.reward_function.compute_reward(pos=np.array(
                [data[2], data[3], data[4]]))  # data[2-4] are the position, from that the reward is computed
            reward += rew

        failure_counter = float(failure_counter)
        # if failure_counter > 0:
        #     print(failure_counter)
        reward += self.constant_penalty
        reward = np.float32(reward)
        obs = [speed, gear, rpm, track_information, acceleration, steering_angle, slipping_tires, crash,
               failure_counter]
        return obs, reward, terminated, info

    def normalize_track(self, l_x, l_z, r_x, r_z, car_position, yaw):
        angle = yaw
        left = (np.array([l_x, l_z]).T - car_position).T
        right = (np.array([r_x, r_z]).T - car_position).T

        left_normal_x = left[0] * np.cos(angle) - left[1] * np.sin(angle)
        left_normal_y = left[0] * np.sin(angle) + left[1] * np.cos(angle)

        right_normal_x = right[0] * np.cos(angle) - right[1] * np.sin(angle)
        right_normal_y = right[0] * np.sin(angle) + right[1] * np.cos(angle)

        return left_normal_x, left_normal_y, right_normal_x, right_normal_y

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data = self.grab_data()
        track_information = np.full((60,), 0, dtype='float32')
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')

        acceleration = np.array([
            data[18],
        ], dtype='float32')
        steering_angle = np.array([
            data[19],
        ], dtype='float32')
        slipping_tires = np.array(data[20:24], dtype='float32')
        crash = np.array([
            data[24],
        ], dtype='float32')
        failure_counter = 0.0
        obs = [speed, gear, rpm, track_information, acceleration, steering_angle, slipping_tires, crash,
               failure_counter]
        self.reward_function.reset()
        return obs, {}

    def get_track_in_front(self, car_position, look_ahead_distance, nearby_correction):
        # Find point that is closest to the car, from all the points, both left and right side
        entire_map = self.map_left.T.tolist() + self.map_right.T.tolist()
        tree = spatial.KDTree(entire_map)
        (_, i) = tree.query(car_position)
        if i < len(self.map_left.T):  # if the closest point is on the left side
            # print("left side is closer")

            i_l = i  # this index is the index for the closest point on the left side of the track
            i_l_min = i_l
            # find the nearest point on the right side of the track, but look for only nearby points
            j_min = max(i_l_min - nearby_correction, 0)  # lower bound
            j_max = min(i_l_min + nearby_correction, len(self.map_left.T) - 1)  # upper bound
            tree_r = spatial.KDTree(
                self.map_right.T[j_min:j_max])  # look up the index of the closest point on the other side of the track
            (_, i_r_min) = tree_r.query(self.map_left.T[i_l_min])
            i_r_min = i_r_min + j_min

            # #calculate the endpoint for the other side of the track
            # j_min = max(i_l+look_ahead_distance-nearby_correction,0) # lower bound
            # j_max = min(i_l+look_ahead_distance+nearby_correction,len(map_left.T)-1) # upper bound
            # tree_r_far = spatial.KDTree(map_right.T[j_min:j_max]) # look up the index of the closest point
            # (_, i_r_max) = tree_r_far.query(map_left.T[i_l_max])
            # i_r_max = i_r_max + j_min

        else:
            # print("right side is closer")
            i_r = i - len(
                self.map_left.T)  # this index is the index for the closest point on the right side of the track
            i_r_min = i_r
            # find the nearest point on the left side of the track, but look for only nearby points
            j_min = max(i_r - nearby_correction, 0)  # lower bound
            j_max = min(i_r + nearby_correction, len(self.map_right.T) - 1)  # upper bound
            tree_l = spatial.KDTree(self.map_left.T[j_min:j_max])  # look up the index of the closest point
            (_, i_l_min) = tree_l.query(self.map_right.T[i_r])
            i_l_min = i_l_min + j_min

        i_l_max = i_l_min + look_ahead_distance
        i_r_max = i_r_min + look_ahead_distance

        extra = np.full((look_ahead_distance, 2), self.map_left.T[-1])
        map_left_extended = np.append(self.map_left.T, extra, axis=0).T

        extra = np.full((look_ahead_distance, 2), self.map_right.T[-1])
        map_right_extended = np.append(self.map_right.T, extra, axis=0).T

        l_x = map_left_extended[0][i_l_min:i_l_max]
        l_z = map_left_extended[1][i_l_min:i_l_max]
        r_x = map_right_extended[0][i_r_min:i_r_max]
        r_z = map_right_extended[1][i_r_min:i_r_max]
        return l_x, l_z, r_x, r_z
