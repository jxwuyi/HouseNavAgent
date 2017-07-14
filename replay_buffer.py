import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

#############  Replay Buffer ##############
class ReplayBuffer(object):
    def __init__(self, size, frame_history_len, frame_type = np.uint8,
                action_shape = [], action_type = np.int32):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.frame_type = frame_type
        self.action_shape = action_shape
        self.action_type = action_type

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

        self.batch_size = None
        self.obs_batch = None
        self.obs_nxt_batch = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        #obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        for i, idx in enumerate(idxes):
            self.obs_batch[i] = self._encode_observation(idx)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        #next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        for i, idx in enumerate(idxes):
            self.obs_nxt_batch[i] = self._encode_observation(idx + 1)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return self.obs_batch, act_batch, rew_batch, self.obs_nxt_batch, done_mask

    def sample(self, batch_size, _idxes=None):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype <self.frame_type>
        act_batch: np.array
            Array of shape (batch_size, <self.action_shape>) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype <self.frame_type>
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size) or (batch_size < 0)
        if _idxes is not None:
            idxes = _idxes
        else:
            if batch_size < 0:
                idxes = list(range(0, self.num_in_buffer - 1))
            else:
                idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)

        if (self.batch_size is None) or (len(idxes) != self.batch_size):
            self.batch_size = len(idxes)
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            self.obs_batch = np.zeros([self.batch_size, self.frame_history_len, img_h, img_w, 3],dtype=self.frame_type)
            self.obs_nxt_batch = np.zeros([self.batch_size, self.frame_history_len, img_h, img_w, 3],dtype=self.frame_type)

        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2 or self.frame_history_len == 1: # when only a single frame, directly return it
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            #frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            frames = np.zeros((self.frame_history_len, img_h, img_w, 3), dtype=self.frame_type)
            pt = missing_context
            for idx in range(start_idx, end_idx):
                #frames.append(self.obs[idx % self.size])
                frames[pt] = self.obs[idx % self.size]
                pt += 1
            #return np.concatenate(frames, 2)
            return frames
        else:
            # this optimization has potential to saves about 30% compute time \o/
            #img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            #return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)
            #frames = []
            #for id in range(start_idx, end_idx):
            #    frames.append(self.obs[id])
            #return np.concatenate(frames, 2)
            return self.obs[start_idx:end_idx]

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=self.frame_type)
            action_shape = list(self.action_shape)
            self.action   = np.empty([self.size] + action_shape,      dtype=self.action_type)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done, info):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        info: extra info, ignored
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done


"""
Replay with extra storage and maintaining partition information (for aux tasks)
"""
class FullReplayBuffer(ReplayBuffer):
    def __init__(self, size, frame_history_len, frame_type = np.uint8,
                action_shape = [], action_type = np.int32, partition=[],
                default_partition = None,
                extra_info_shapes=[], extra_info_types=[]):
        """
        @param parition:
           a list of tuple (n_parition, partition_function)
           partition_function(info) --> 0...n_partition-1
        """
        super(FullReplayBuffer, self).__init__(size, frame_history_len, frame_type, action_shape, action_type)
        self.infos = [None] * self.size
        self.n_part = len(partition)
        if self.n_part > 0:
            self.part_pos = -1 * np.ones([size, self.n_part, 2], dtype=np.int32)  # partition, part_pos
        else:
            self.part_pos = None
        self.partition = [[[] for _ in range(p[0])] for p in partition]
        self.partition_func = [p[1] for p in partition]
        self.default_partition = default_partition
        self.extra_info_shapes = extra_info_shapes
        self.extra_info_types = extra_info_types
        self.extra_infos = None
        assert len(self.extra_info_shapes) == len(self.extra_info_types), \
            '[FullReplayBuffer] Lengths of <extra_info_shapes> and <extra_info_types> must match! Now received {} and {}'.format(extra_info_shapes, extra_info_types)

    def _remove_part_index(self, idx, part_pos): # remove partition information
        for i in range(self.n_part):
            k, p = part_pos[i]
            partition = self.partition[i]
            t = partition[k].pop()
            if t == idx:
                continue
            partition[k][p] = t
            self.part_pos[t,i,1] = p

    def _add_part_index(self, idx, part_pos): # add partition information
        for i, k in enumerate(part_pos):
            self.part_pos[idx,i,0]=k
            partition = self.partition[i]
            self.part_pos[idx,i,1]=len(partition[k])
            partition[k].append(idx)

    def _add_extra_info(self, idx, info):
        if self.extra_infos is None:
            self.extra_infos = []
            for i in range(len(info)):
                if isinstance(info[i], int):
                    cur = np.empty([self.size], dtype=np.int32)
                else:
                    tp = np.float32 if len(self.extra_info_types) <= i else self.extra_info_types[i]
                    sp = info[i].shape if len(self.extra_info_shapes) <= i else self.extra_info_shapes[i]
                    cur = np.empty([self.size] + list(sp), dtype=tp)
                self.extra_infos.append(cur)
        for i, dat in enumerate(info):
            self.extra_infos[i][idx] = dat

    def store_effect(self, idx, action, reward, done, info, extra_infos=None):
        super(FullReplayBuffer, self).store_effect(idx, action, reward, done, None)
        self.infos[idx] = info
        if self.part_pos is not None:
            if self.part_pos[idx,0,0] > -1:  # remove the previous instance from partitions
                self._remove_part_index(idx, self.part_pos[idx])
            new_index = [func(info) for func in self.partition_func]
            self._add_part_index(idx, new_index)
        if extra_infos is not None:
            if not isinstance(extra_infos, list):
                extra_infos = list(extra_infos)
            self._add_extra_info(idx, extra_infos)

    def sample(self, batch_size, partition=None, partition_sampler=None, collect_info=None,
               collect_extras=False, collect_extra_next=False):
        assert batch_size > 0, '[FullReplayBuffer] Currently only support sample for batch_size > 0'
        if partition is None: partition = self.default_partition
        if partition is None:  # uniformly sample
            idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        else:
            assert isinstance(partition, int), '[FullReplayBuffer] partition must be an <int>, the index of the specified partition'
            k = partition
            cur_partition = self.partition[k]
            if partition_sampler is None:
                active_chunk = [ch for ch in cur_partition if len(ch) > 0]
                n_active = len(active_chunk)
                def sampler():
                    while True:
                        chunk = active_chunk[np.random.choice(n_active)]
                        n = len(chunk)
                        ret = chunk[np.random.choice(n)]
                        if ret < self.num_in_buffer - 1:
                            return ret
            else:
                def sampler():
                    while True:
                        i = partition_sampler()
                        cur_part = cur_partition[i]
                        if len(cur_part) > 0:
                            m = len(cur_part)
                            ret = cur_part[np.random.choice(m)]
                            if ret < self.num_in_buffer - 1:
                                return ret
            idxes = sample_n_unique(sampler, batch_size)
            #idxes = [sampler() for _ in range(batch_size)]  # allow same samples
        self._idxes = idxes
        extras = []
        if collect_info is not None:
            extras.append([collect_info(self.infos[idx]) for idx in idxes])
        if collect_extras:
            extras.append([ex[idxes] for ex in self.extra_infos])
        if collect_extra_next:
            next_idxes = [idx + 1 for idx in idxes]
            extras.append([ex[next_idxes] for ex in self.extra_infos])
        ret_vals = list(super(FullReplayBuffer, self).sample(batch_size, _idxes=idxes))
        return ret_vals + extras


#########################################################
# Replay Buffer for Recurrent Neural Net
class RNNReplayBuffer(object):
    def __init__(self, size, max_seq_len, frame_type = np.uint8,
                 action_shape = [], action_type = np.int32):
        """This is a replay buffer for recurrent networks.

        The sepecific memory optimizations use here are:
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        Warning! Assumes that returning frame of zeros at the end
        of the episode, when there are less frames than `max_seq_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        max_seq_len: int
            Number of frames per episode.
        """
        self.size = size
        self.max_seq_len = max_seq_len
        self.frame_type = frame_type
        self.action_shape = action_shape
        self.action_type = action_type

        self.next_idx      = 0
        self.next_frame    = 0
        self.num_in_buffer = 0
        self.total_samples = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

        self.batch_size = None
        self.seq_len = None
        self.obs_batch = None
        self.obs_epis = None

        self.recent_frame = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, seq_len):
        #obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        total_length = 0
        for i, idx in enumerate(idxes):
            self.msk_batch[i] = 0
            _len, done, _obs_epis, _act_epis, _rew_epis = self._encode_observation(idx, seq_len)
            start_pos = _len // 2    # Only compute loss of the last half of the episode
            total_length += _len - start_pos
            self.msk_batch[i, start_pos:_len] = 1
            self.done_batch[i, start_pos:_len] = 1
            if done: self.done_batch[i, _len - 1] = 0
            self.obs_batch[i] = _obs_epis
            self.act_batch[i] = _act_epis
            self.rew_batch[i] = _rew_epis

        return self.obs_batch, self.act_batch, self.rew_batch, self.msk_batch, self.done_batch, total_length

    def sample(self, batch_size, seq_len = None):
        """
        return samples of shape (seq_len, batch_size, *shapes)
        """
        assert self.can_sample(batch_size) or (batch_size < 0)
        if seq_len is None: seq_len = self.max_seq_len
        if batch_size < 0:
            idxes = list(range(0, self.num_in_buffer))
        else:
            idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 1), batch_size)

        if (self.batch_size is None) or (len(idxes) != self.batch_size) \
            or (seq_len != self.seq_len):
            self.batch_size = len(idxes)
            self.seq_len = seq_len
            img_h, img_w, img_c = self.recent_frame.shape
            self.obs_batch = np.zeros([self.batch_size, seq_len+1, img_h, img_w, img_c],dtype=self.frame_type)
            self.obs_epis  = np.zeros([seq_len+1, img_h, img_w, img_c],dtype=self.frame_type)
            self.act_batch = np.zeros([self.batch_size, seq_len] + self.action_shape, dtype=self.action_type)
            self.act_epis  = np.zeros([seq_len] + self.action_shape,dtype=self.frame_type)
            self.rew_batch = np.zeros([self.batch_size, seq_len], dtype=np.float32)
            self.rew_epis  = np.zeros([seq_len], dtype=np.float32)
            self.msk_batch = np.zeros([self.batch_size, seq_len], dtype=np.uint8)
            self.done_batch= np.zeros([self.batch_size, seq_len], dtype=np.uint8)

        return self._encode_sample(idxes, seq_len)

    def encode_recent_observation(self):
        """Return the most recent frame (img_c, img_h, img_w)."""
        assert self.recent_frame is not None
        return self.recent_frame

    def _encode_observation(self, idx, seq_len):
        """
        return (length, done, frames, actions, reward) for an episode
        """
        # pick range: [start_idx, end_idx), end_idx is exclusive
        total_len = self.lengths[idx]
        if total_len <= seq_len:
            start_idx, end_idx = 0, total_len
        else:
            upper_bound = total_len - seq_len + 1 # min(total_len - seq_len, seq_len)
            # upper_bound = total_len - seq_len
            start_idx = np.random.choice(upper_bound)
            end_idx = start_idx + seq_len
        cur_len = end_idx - start_idx
        # clear frames
        self.act_epis[end_idx:, ...]=0
        self.rew_epis[end_idx:]=0

        # fill zeros
        # self.obs_epis[end_idx:, ...]=0

        # fill frames
        done = (end_idx == total_len)
        redudant_frames = seq_len + 1 - cur_len
        if not done:
            self.obs_epis[cur_len] = self.obs[idx, end_idx]
            redudant_frames -= 1
        if redudant_frames > 0:
            redudant_idx = [np.random.choice(total_len) for _ in range(redudant_frames)]
            self.obs_epis[-redudant_frames:] = self.obs[idx, redudant_idx]

        self.obs_epis[:cur_len] = self.obs[idx, start_idx:end_idx]
        self.act_epis[:cur_len] = self.action[idx, start_idx:end_idx]
        self.rew_epis[:cur_len] = self.reward[idx, start_idx:end_idx]
        return cur_len, done, self.obs_epis, self.act_epis, self.rew_epis

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.next_frame < 0:
            self.total_samples -= self.lengths[self.next_idx]
            self.lengths[self.next_idx] = 0
            self.next_frame = 0
        assert (self.next_frame < self.max_seq_len)
        self.recent_frame = frame
        if self.obs is None:
            self.obs      = np.empty([self.size, self.max_seq_len] + list(frame.shape), dtype=self.frame_type)
            action_shape = list(self.action_shape)
            self.action   = np.empty([self.size, self.max_seq_len] + action_shape, dtype=self.action_type)
            self.reward   = np.empty([self.size, self.max_seq_len], dtype=np.float32)
            self.lengths  = np.zeros([self.size], dtype=np.int32)
        self.obs[self.next_idx, self.next_frame] = frame

        ret = (self.next_idx, self.next_frame)
        self.next_frame += 1
        self.total_samples += 1
        return ret

    def store_effect(self, idx, action, reward, done, info):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: (int, int)
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        ep, fr = idx
        self.action[ep, fr] = action
        self.reward[ep, fr] = reward
        self.lengths[ep] += 1
        if done:
            self.next_idx = (self.next_idx + 1) % self.size
            self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
            self.next_frame = -1
