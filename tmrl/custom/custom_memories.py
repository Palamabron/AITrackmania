import random
import numpy as np

from tmrl.memory import TorchMemory, R2D2Memory


# LOCAL BUFFER COMPRESSION ==============================

def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1], obs[2][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_mobilenet(prev_act, obs, rew, terminated, truncated, info):
    '''
    Prepares received data for storage in a local buffer
    Returns:
    A tuple containing modified versions of the received data (prev_act, obs, rew, terminated, truncated, info).
    Functionality:
    The function converts the reward rew into a NumPy float32 (rew_mod = np.float32(rew)).
    '''
    obs_mod = obs
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """
    Sample compressor for MemoryTMFull
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    prev_act_mod = prev_act
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod


# FUNCTIONS ====================================================


def last_true_in_list(li):
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist, eoe_idx_in_hist):
    """
    Pads the history hist before the End Of Episode (EOE) index.

    Previous entries in hist are padded with copies of the first element occurring after EOE.
    """
    last_idx = len(hist) - 1
    assert eoe_idx_in_hist <= last_idx, f"replace_hist_before_eoe: eoe_idx_in_hist:{eoe_idx_in_hist}, last_idx:{last_idx}"
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


# SUPPORTED CUSTOM MEMORIES ============================================================================================


class GenericTorchMemory(TorchMemory):
    def __init__(self,
                 memory_size=1e6,
                 batch_size=1,
                 dataset_path="",
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):

        # parse:
        d0 = [b[0] for b in buffer.memory]  # actions
        d1 = [b[1] for b in buffer.memory]  # observations
        d2 = [b[2] for b in buffer.memory]  # rewards
        d3 = [b[3] for b in buffer.memory]  # terminated
        d4 = [b[4] for b in buffer.memory]  # truncated
        d5 = [b[5] for b in buffer.memory]  # info
        d6 = [b[3] or b[4] for b in buffer.memory]  # done

        # append:
        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        # trim
        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):

        # This is a hack to avoid invalid transitions from terminal to initial
        # TODO: find a way to only index valid transitions instead
        while self.data[6][item]:
            item = random.randint(a=0, b=self.__len__() - 1)

        idx_last = item
        idx_now = item + 1

        last_obs = self.data[1][idx_last]
        new_act = self.data[0][idx_now]
        rew = self.data[2][idx_now]
        new_obs = self.data[1][idx_now]
        terminated = self.data[3][idx_now]
        truncated = self.data[4][idx_now]
        info = self.data[5][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


class MemoryTM(TorchMemory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)
        '''
        Arguments:

        memory_size: Maximum size of the memory buffer.
        batch_size: Size of batches used during training.
        dataset_path: Path to the dataset.
        imgs_obs: Number of observed images.
        act_buf_len: Length of the action buffer.
        nb_steps: Number of steps.
        sample_preprocessor: A callable function for sample preprocessing.
        crc_debug: Flag indicating whether to debug CRC (Cyclic Redundancy Check).
        device: Device where the memory is stored (e.g., "cpu" or "cuda").
        '''

    def append_buffer(self, buffer):
        '''
        This function is intended to be implemented in subclasses but is not implemented in the MemoryTM class itself.
        Raises a NotImplementedError, prompting subclasses to implement this method based on specific requirements.
        '''
        raise NotImplementedError

    def __len__(self):
        '''
        Calculates the length of the memory buffer by considering the number of samples stored in the memory.
        If there is no data in the memory buffer, returns 0. Otherwise, returns the length of the data minus min_samples minus 1.
        '''
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        '''
        Similar to append_buffer, this function is intended to be implemented in subclasses but is not implemented in the MemoryTM class itself.
        Raises a NotImplementedError, prompting subclasses to implement this method based on specific requirements.
        '''
        raise NotImplementedError


class MemoryTMLidar(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[7][idx_now]
        truncated = self.data[8][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes (terminated or truncated)
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[3] for b in buffer.memory]  # terminated
        d8 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]

        return self


class MemoryTMLidarProgress(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[8][idx_now]
        truncated = self.data[9][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        '''
        Arguments:

        item: Index used to load lidar images from stored data.
        Functionality:

        Loads a sequence of lidar images from the stored data, starting from the given index (item) with a specific number of observed lidar images (imgs_obs).
        '''
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        '''
        Arguments:

        item: Index used to load actions from stored data.
        Functionality:

        Loads a sequence of actions from the stored data, starting from the given index (item) with a specific length of action buffer (act_buf_len).
        '''
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, truncated, terminated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][2] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # progress
        d8 = [b[3] for b in buffer.memory]  # terminated
        d9 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]

        return self


class MemoryTMFull(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        last_obs = (
            self.data[2][idx_last],
            self.data[7][idx_last],
            self.data[8][idx_last],
            imgs_last_obs,
            *last_act_buf
        )
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][3] for b in buffer.memory]  # images
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # gears
        d8 = [b[1][2] for b in buffer.memory]  # rpms
        d9 = [b[3] for b in buffer.memory]  # terminated
        d10 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self


# ============= custom mobilenet memory ==============

class MemoryTMBest(MemoryTM):

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[27][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        last_obs = (
            self.data[2][idx_last],  # 0
            self.data[3][idx_last],  # 1
            self.data[4][idx_last],  # 2
            self.data[5][idx_last],  # 3
            self.data[6][idx_last],  # 4
            self.data[7][idx_last],  # 5
            self.data[8][idx_last],  # 6
            self.data[9][idx_last],  # 7
            self.data[10][idx_last],  # 8
            self.data[11][idx_last],  # 9
            self.data[12][idx_last],  # 10
            self.data[13][idx_last],  # 11
            self.data[14][idx_last],  # 12
            self.data[15][idx_last],  # 13
            self.data[16][idx_last],  # 14
            self.data[17][idx_last],  # 15
            self.data[18][idx_last],  # 16
            self.data[19][idx_last],  # 17
            self.data[20][idx_last],  # 18
            self.data[21][idx_last],  # 19
            self.data[22][idx_last],  # 20
            self.data[23][idx_last],  # 21
            self.data[24][idx_last],  # 22
            self.data[25][idx_last],  # 23
            self.data[26][idx_last],  # 24 imgs
            # imgs_last_obs,
            *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[28][idx_now])
        new_obs = (
            self.data[2][idx_now],  # 0
            self.data[3][idx_now],  # 1
            self.data[4][idx_now],  # 2
            self.data[5][idx_now],  # 3
            self.data[6][idx_now],  # 4
            self.data[7][idx_now],  # 5
            self.data[8][idx_now],  # 6
            self.data[9][idx_now],  # 7
            self.data[10][idx_now],  # 8
            self.data[11][idx_now],  # 9
            self.data[12][idx_now],  # 10
            self.data[13][idx_now],  # 11
            self.data[14][idx_now],  # 12
            self.data[15][idx_now],  # 13
            self.data[16][idx_now],  # 14
            self.data[17][idx_now],  # 15
            self.data[18][idx_now],  # 16
            self.data[19][idx_now],  # 17
            self.data[20][idx_now],  # 18
            self.data[21][idx_now],  # 19
            self.data[22][idx_now],  # 20
            self.data[23][idx_now],  # 21
            self.data[24][idx_now],  # 22
            self.data[25][idx_now],  # 23
            self.data[26][idx_now],  # 24 imgs
            # imgs_new_obs,
            *new_act_buf)
        terminated = self.data[30][idx_now]
        truncated = self.data[31][idx_now]
        info = self.data[29][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[26][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        # line 532, in get_transition
        # self.data[23][idx_last],  # 21
        # IndexError: list index out of range
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [np.array([b[1][0]]) for b in buffer.memory]  # positions
        d3 = [np.array([b[1][1]]) for b in buffer.memory]  # speeds
        d4 = [np.array([b[1][2]]) for b in buffer.memory]  # acceleration
        d5 = [np.array([b[1][3]]) for b in buffer.memory]  # jerk
        d6 = [np.array([b[1][4]]) for b in buffer.memory]  # race_progress
        d7 = [np.array([b[1][5]]) for b in buffer.memory]  # input_steer
        d8 = [np.array([b[1][6]]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array([b[1][7]]) for b in buffer.memory]  # input_brake
        d10 = [np.array([b[1][8]]) for b in buffer.memory]  # gear
        d11 = [np.array([b[1][9]]) for b in buffer.memory]  # rpm
        d12 = [np.array([b[1][10]]) for b in buffer.memory]  # aim_yaw
        d13 = [np.array([b[1][11]]) for b in buffer.memory]  # aim_pitch
        d14 = [np.array([b[1][12][0]]) for b in buffer.memory]  # surface_id
        d15 = [np.array([b[1][13][0]]) for b in buffer.memory]  # steer_angle
        d16 = [np.array([b[1][14][0]]) for b in buffer.memory]  # wheel_rot
        d17 = [np.array([b[1][15][0]]) for b in buffer.memory]  # wheel_rot_speed
        d18 = [np.array(b[1][16]) for b in buffer.memory]  # damper_len
        d19 = [np.array(b[1][17]) for b in buffer.memory]  # slip_coef
        d20 = [np.array([b[1][18]]) for b in buffer.memory]  # reactor_ground_mode
        d21 = [np.array([b[1][19]]) for b in buffer.memory]  # ground_contact
        d22 = [np.array(b[1][20]) for b in buffer.memory]  # reactor_air_control
        d23 = [np.array([b[1][21]]) for b in buffer.memory]  # ground_dist
        d24 = [b[1][22].tolist() for b in buffer.memory]  # crashed
        d24 = [np.array([el]) for el in d24]
        d25 = [np.array([b[1][23]]) for b in buffer.memory]  # failure counter
        d26 = [b[1][24] for b in buffer.memory]  # imgs

        d27 = [b[3] or b[4] for b in buffer.memory]  # eoes (end of episode)
        d28 = [b[2] for b in buffer.memory]  # rewards
        d29 = [b[5] for b in buffer.memory]  # infos
        d30 = [b[3] for b in buffer.memory]  # terminated
        d31 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
            self.data[11] += d11
            self.data[12] += d12
            self.data[13] += d13
            self.data[14] += d14
            self.data[15] += d15
            self.data[16] += d16
            self.data[17] += d17
            self.data[18] += d18
            self.data[19] += d19
            self.data[20] += d20
            self.data[21] += d21
            self.data[22] += d22
            self.data[23] += d23
            self.data[24] += d24
            self.data[25] += d25
            self.data[26] += d26
            self.data[27] += d27
            self.data[28] += d28
            self.data[29] += d29
            self.data[30] += d30
            self.data[31] += d31

        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)
            self.data.append(d11)
            self.data.append(d12)
            self.data.append(d13)
            self.data.append(d14)
            self.data.append(d15)
            self.data.append(d16)
            self.data.append(d17)
            self.data.append(d18)
            self.data.append(d19)
            self.data.append(d20)
            self.data.append(d21)
            self.data.append(d22)
            self.data.append(d23)
            self.data.append(d24)
            self.data.append(d25)
            self.data.append(d26)
            self.data.append(d27)
            self.data.append(d28)
            self.data.append(d29)
            self.data.append(d30)
            self.data.append(d31)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
            self.data[11] = self.data[11][to_trim:]
            self.data[12] = self.data[12][to_trim:]
            self.data[13] = self.data[13][to_trim:]
            self.data[14] = self.data[14][to_trim:]
            self.data[15] = self.data[15][to_trim:]
            self.data[16] = self.data[16][to_trim:]
            self.data[17] = self.data[17][to_trim:]
            self.data[18] = self.data[18][to_trim:]
            self.data[19] = self.data[19][to_trim:]
            self.data[20] = self.data[20][to_trim:]
            self.data[21] = self.data[21][to_trim:]
            self.data[22] = self.data[22][to_trim:]
            self.data[23] = self.data[23][to_trim:]
            self.data[24] = self.data[24][to_trim:]
            self.data[25] = self.data[25][to_trim:]
            self.data[26] = self.data[26][to_trim:]
            self.data[27] = self.data[27][to_trim:]
            self.data[28] = self.data[28][to_trim:]
            self.data[29] = self.data[29][to_trim:]
            self.data[30] = self.data[30][to_trim:]
            self.data[31] = self.data[31][to_trim:]

        return self


# ============================= R2D2 MEMORY ===============================

class MemoryR2D2(R2D2Memory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=2,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[17][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        last_obs = (
            self.data[2][idx_last],  # 0
            self.data[3][idx_last],  # 1
            self.data[4][idx_last],  # 2
            self.data[5][idx_last],  # 3
            self.data[6][idx_last],  # 4
            self.data[7][idx_last],  # 5
            self.data[8][idx_last],  # 6
            self.data[9][idx_last],  # 7
            self.data[10][idx_last],  # 8
            self.data[11][idx_last],  # 9
            self.data[12][idx_last],  # 10
            self.data[13][idx_last],  # 11
            self.data[14][idx_last],  # 12
            self.data[15][idx_last],  # 13
            self.data[16][idx_last],  # 14
            # imgs_last_obs,
            *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[18][idx_now])
        new_obs = (
            self.data[2][idx_now],  # 0
            self.data[3][idx_now],  # 1
            self.data[4][idx_now],  # 2
            self.data[5][idx_now],  # 3
            self.data[6][idx_now],  # 4
            self.data[7][idx_now],  # 5
            self.data[8][idx_now],  # 6
            self.data[9][idx_now],  # 7
            self.data[10][idx_now],  # 8
            self.data[11][idx_now],  # 9
            self.data[12][idx_now],  # 10
            self.data[13][idx_now],  # 11
            self.data[14][idx_now],  # 12
            self.data[15][idx_now],  # 13
            self.data[16][idx_now],  # 14
            # imgs_new_obs,
            *new_act_buf)
        terminated = self.data[20][idx_now]
        truncated = self.data[21][idx_now]
        info = self.data[19][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[16][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        # line 532, in get_transition
        # self.data[23][idx_last],  # 21
        # IndexError: list index out of range
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [np.array(b[1][0]) for b in buffer.memory]  # checkpoints
        d3 = [np.array(b[1][1]) for b in buffer.memory]  # speeds
        d4 = [np.array(b[1][2]) for b in buffer.memory]  # accelerations
        d5 = [np.array(b[1][3]) for b in buffer.memory]  # jerks
        d6 = [np.array(b[1][4]) for b in buffer.memory]  # race_progress
        d7 = [np.array(b[1][5]) for b in buffer.memory]  # input_steer
        d8 = [np.array(b[1][6]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array(b[1][7]) for b in buffer.memory]  # input_brake
        d10 = [np.array(b[1][8]) for b in buffer.memory]  # gear
        d11 = [np.array(b[1][9]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array(b[1][10]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array(b[1][11]) for b in buffer.memory]  # steer_angle
        d14 = [np.array(b[1][12]) for b in buffer.memory]  # slip_coef
        d15 = [np.array(b[1][13]) for b in buffer.memory]  # failure counter
        d16 = [np.array(b[1][14]) for b in buffer.memory]  # imgs
        d17 = [b[3] or b[4] for b in buffer.memory]  # eoes (end of episode)
        d18 = [b[2] for b in buffer.memory]  # rewards
        d19 = [b[5] for b in buffer.memory]  # infos
        d20 = [b[3] for b in buffer.memory]  # terminated
        d21 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
            self.data[11] += d11
            self.data[12] += d12
            self.data[13] += d13
            self.data[14] += d14
            self.data[15] += d15
            self.data[16] += d16
            self.data[17] += d17
            self.data[18] += d18
            self.data[19] += d19
            self.data[20] += d20
            self.data[21] += d21

        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)
            self.data.append(d11)
            self.data.append(d12)
            self.data.append(d13)
            self.data.append(d14)
            self.data.append(d15)
            self.data.append(d16)
            self.data.append(d17)
            self.data.append(d18)
            self.data.append(d19)
            self.data.append(d20)
            self.data.append(d21)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
            self.data[11] = self.data[11][to_trim:]
            self.data[12] = self.data[12][to_trim:]
            self.data[13] = self.data[13][to_trim:]
            self.data[14] = self.data[14][to_trim:]
            self.data[15] = self.data[15][to_trim:]
            self.data[16] = self.data[16][to_trim:]
            self.data[17] = self.data[17][to_trim:]
            self.data[18] = self.data[18][to_trim:]
            self.data[19] = self.data[19][to_trim:]
            self.data[20] = self.data[20][to_trim:]
            self.data[21] = self.data[21][to_trim:]

        return self


# ============================ R2D2 MEMORY without Images ==================

class MemoryR2D2woImages(R2D2Memory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=2,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]


        last_obs = (
            self.data[2][idx_last],  # 0
            self.data[3][idx_last],  # 1
            self.data[4][idx_last],  # 2
            self.data[5][idx_last],  # 3
            self.data[6][idx_last],  # 4
            self.data[7][idx_last],  # 5
            self.data[8][idx_last],  # 6
            self.data[9][idx_last],  # 7
            self.data[10][idx_last],  # 8
            self.data[11][idx_last],  # 9
            self.data[12][idx_last],  # 10
            self.data[13][idx_last],  # 11
            self.data[14][idx_last],  # 12
            self.data[15][idx_last],  # 13
            *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[17][idx_now])
        new_obs = (
            self.data[2][idx_now],  # 0
            self.data[3][idx_now],  # 1
            self.data[4][idx_now],  # 2
            self.data[5][idx_now],  # 3
            self.data[6][idx_now],  # 4
            self.data[7][idx_now],  # 5
            self.data[8][idx_now],  # 6
            self.data[9][idx_now],  # 7
            self.data[10][idx_now],  # 8
            self.data[11][idx_now],  # 9
            self.data[12][idx_now],  # 10
            self.data[13][idx_now],  # 11
            self.data[14][idx_now],  # 12
            self.data[15][idx_now],  # 13
            *new_act_buf)
        terminated = self.data[19][idx_now]
        truncated = self.data[20][idx_now]
        info = self.data[18][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[2][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        # line 532, in get_transition
        # self.data[23][idx_last],  # 21
        # IndexError: list index out of range
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [np.array(b[1][0]) for b in buffer.memory]  # checkpoints
        d3 = [np.array(b[1][1]) for b in buffer.memory]  # speeds
        d4 = [np.array(b[1][2]) for b in buffer.memory]  # accelerations
        d5 = [np.array(b[1][3]) for b in buffer.memory]  # jerks
        d6 = [np.array(b[1][4]) for b in buffer.memory]  # race_progress
        d7 = [np.array(b[1][5]) for b in buffer.memory]  # input_steer
        d8 = [np.array(b[1][6]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array(b[1][7]) for b in buffer.memory]  # input_brake
        d10 = [np.array(b[1][8]) for b in buffer.memory]  # gear
        d11 = [np.array(b[1][9]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array(b[1][10]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array(b[1][11]) for b in buffer.memory]  # steer_angle
        d14 = [np.array(b[1][12]) for b in buffer.memory]  # slip_coef
        d15 = [np.array(b[1][13]) for b in buffer.memory]  # failure counter
        d16 = [b[3] or b[4] for b in buffer.memory]  # eoes (end of episode)
        d17 = [b[2] for b in buffer.memory]  # rewards
        d18 = [b[5] for b in buffer.memory]  # infos
        d19 = [b[3] for b in buffer.memory]  # terminated
        d20 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
            self.data[11] += d11
            self.data[12] += d12
            self.data[13] += d13
            self.data[14] += d14
            self.data[15] += d15
            self.data[16] += d16
            self.data[17] += d17
            self.data[18] += d18
            self.data[19] += d19
            self.data[20] += d20

        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)
            self.data.append(d11)
            self.data.append(d12)
            self.data.append(d13)
            self.data.append(d14)
            self.data.append(d15)
            self.data.append(d16)
            self.data.append(d17)
            self.data.append(d18)
            self.data.append(d19)
            self.data.append(d20)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
            self.data[11] = self.data[11][to_trim:]
            self.data[12] = self.data[12][to_trim:]
            self.data[13] = self.data[13][to_trim:]
            self.data[14] = self.data[14][to_trim:]
            self.data[15] = self.data[15][to_trim:]
            self.data[16] = self.data[16][to_trim:]
            self.data[17] = self.data[17][to_trim:]
            self.data[18] = self.data[18][to_trim:]
            self.data[19] = self.data[19][to_trim:]
            self.data[20] = self.data[20][to_trim:]

        return self


# ============================ R2D2 MEMORY without Images ==================

class MemoryR2D2Sophy(R2D2Memory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=2,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        last_obs = (
            self.data[2][idx_last],  # 0
            self.data[3][idx_last],  # 1
            self.data[4][idx_last],  # 2
            self.data[5][idx_last],  # 3
            self.data[6][idx_last],  # 4
            self.data[7][idx_last],  # 5
            self.data[8][idx_last],  # 6
            self.data[9][idx_last],  # 7
            self.data[10][idx_last],  # 8
            self.data[11][idx_last],  # 9
            self.data[12][idx_last],  # 10
            self.data[13][idx_last],  # 11
            self.data[14][idx_last],  # 12
            self.data[15][idx_last],  # 13
            *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[17][idx_now])
        new_obs = (
            self.data[2][idx_now],  # 0
            self.data[3][idx_now],  # 1
            self.data[4][idx_now],  # 2
            self.data[5][idx_now],  # 3
            self.data[6][idx_now],  # 4
            self.data[7][idx_now],  # 5
            self.data[8][idx_now],  # 6
            self.data[9][idx_now],  # 7
            self.data[10][idx_now],  # 8
            self.data[11][idx_now],  # 9
            self.data[12][idx_now],  # 10
            self.data[13][idx_now],  # 11
            self.data[14][idx_now],  # 12
            self.data[15][idx_now],  # 13
            *new_act_buf)
        terminated = self.data[19][idx_now]
        truncated = self.data[20][idx_now]
        info = self.data[18][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[2][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        # line 532, in get_transition
        # self.data[23][idx_last],  # 21
        # IndexError: list index out of range
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [np.array(b[1][0]) for b in buffer.memory]  # track info
        d3 = [np.array(b[1][1]) for b in buffer.memory]  # speeds
        d4 = [np.array(b[1][2]) for b in buffer.memory]  # accelerations
        d5 = [np.array(b[1][3]) for b in buffer.memory]  # jerks
        d6 = [np.array(b[1][4]) for b in buffer.memory]  # race_progress
        d7 = [np.array(b[1][5]) for b in buffer.memory]  # input_steer
        d8 = [np.array(b[1][6]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array(b[1][7]) for b in buffer.memory]  # input_brake
        d10 = [np.array(b[1][8]) for b in buffer.memory]  # gear
        d11 = [np.array(b[1][9]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array(b[1][10]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array(b[1][11]) for b in buffer.memory]  # steer_angle
        d14 = [np.array(b[1][12]) for b in buffer.memory]  # slip_coef
        d15 = [np.array(b[1][13]) for b in buffer.memory]  # failure counter
        d16 = [b[3] or b[4] for b in buffer.memory]  # eoes (end of episode)
        d17 = [b[2] for b in buffer.memory]  # rewards
        d18 = [b[5] for b in buffer.memory]  # infos
        d19 = [b[3] for b in buffer.memory]  # terminated
        d20 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
            self.data[11] += d11
            self.data[12] += d12
            self.data[13] += d13
            self.data[14] += d14
            self.data[15] += d15
            self.data[16] += d16
            self.data[17] += d17
            self.data[18] += d18
            self.data[19] += d19
            self.data[20] += d20

        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)
            self.data.append(d11)
            self.data.append(d12)
            self.data.append(d13)
            self.data.append(d14)
            self.data.append(d15)
            self.data.append(d16)
            self.data.append(d17)
            self.data.append(d18)
            self.data.append(d19)
            self.data.append(d20)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]
            self.data[11] = self.data[11][to_trim:]
            self.data[12] = self.data[12][to_trim:]
            self.data[13] = self.data[13][to_trim:]
            self.data[14] = self.data[14][to_trim:]
            self.data[15] = self.data[15][to_trim:]
            self.data[16] = self.data[16][to_trim:]
            self.data[17] = self.data[17][to_trim:]
            self.data[18] = self.data[18][to_trim:]
            self.data[19] = self.data[19][to_trim:]
            self.data[20] = self.data[20][to_trim:]

        return self
