from headers import *
from utils import *
from replay_buffer import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class NOPTrainer(AgentTrainer):
    def __init__(self, name, policy, obs_shape, act_shape, args):
        super(NOPTrainer, self).__init__()
        self.name = name
        self.policy = policy  # NOTE: policy must be instantiated before
        assert isinstance(policy, torch.nn.Module), 'policy must be an instantiated instance of torch.nn.Module'
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        self.replay_buffer = ReplayBuffer(
                                50,  # a random small number
                                args['frame_history_len'],
                                action_shape=[self.act_dim],
                                action_type=np.float32)
        self.args = args

    def action(self, gumbel_noise=None):
        frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
        raw_x = self._process_frames(frames, volatile=True)
        batched_actions = self.policy(raw_x)
        if use_cuda:
            cpu_actions = [a.cpu() for a in batched_actions]
        else:
            cpu_actions = batched_actions
        return [a[0].data.numpy() for a in cpu_actions]

    def process_observation(self, obs):
        idx = self.replay_buffer.store_frame(obs)
        return idx

    def process_experience(self, idx, act, rew, done, terminal, info):
        pass

    def preupdate(self):
        pass

    def update(self):
        return None
