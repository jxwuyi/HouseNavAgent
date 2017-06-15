import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

####### Util Functions ############
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def split_batched_array(action, shape):
    actions = []
    p = 0
    for d in shape:
        actions.append(action[:, p:(p + d)])
        p += d
    return actions


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def clip_grad_norm(parameters, max_norm):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    for p in parameters:
        param_norm = p.grad.data.norm()
        clip_coef = max_norm / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)


############ Weight Initialization ############
def initialize_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, math.sqrt(1. / m.in_features))
            #m.weight.data.linear_(-3e-4, 3e-4)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


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

    def sample(self, batch_size):
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

    def store_effect(self, idx, action, reward, done):
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
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done


######## Logging Utils ############
class MyLogger:
    def __init__(self, logdir, clear_file = False, filename = None):
        import os
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.fname = logdir + '/' + (filename or 'progress.txt')
        if clear_file:
            try:
                os.remove(self.fname)
            except OSError:
                pass

    def print(self, str, to_screen = True):
        if to_screen:
            print(str)
        with open(self.fname, 'a') as f:
            print(str, file=f)

def log_var_stats(logger, v):
    logger.print('  -> Param<{}>, '.format(v.size())+\
                 'Val Stats = [norm = %.7f, mean = %.7f, var = %.7f, min = %.7f, max = %.7f]'%(
                    v.data.norm(), v.data.mean(), v.data.var(), v.data.min(), v.data.max()), False)
    if v.grad is None:
        logger.print('              >> Grad Stats = None', False)
    else:
        g = v.grad
        logger.print('              >> Grad Stats = [norm = %.7f, mean = %.7f, var = %.7f, min = %.7f, max = %.7f]' % (
            g.data.norm(), g.data.mean(), g.data.var(), g.data.min(), g.data.max()
        ), False)

def log_parameter_stats(logger, p):
    assert isinstance(p, nn.Module), '[Error in <utils.log_parameter_stats>] policy must be an instance of <nn.Module>'
    assert isinstance(logger, MyLogger), '[Error in <utils.log_parameter_stats>] logger must be an instance of <utils.MyLogger>'
    if hasattr(p,'conv_layers'):
        for i,conv in enumerate(p.conv_layers):
            if conv is None: continue
            logger.print('>> Conv Layer#{} <{}>'.format(i,conv),False)
            for v in conv.parameters():
                log_var_stats(logger, v)
    if hasattr(p,'bc_layers'):
        for i,bc in enumerate(p.bc_layers):
            if bc is None: continue
            logger.print('>> Batch Norm Layer#{} <{}>'.format(i,bc),False)
            for v in bc.parameters():
                log_var_stats(logger, v)
    if hasattr(p,'linear_layers'):
        for i,lr in enumerate(p.linear_layers):
            if lr is None: continue
            logger.print('>> Linear Layer#{} <{}>'.format(i,lr),False)
            for v in lr.parameters():
                log_var_stats(logger, v)
