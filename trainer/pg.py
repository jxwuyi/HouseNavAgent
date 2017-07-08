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

class PolicyGradientTrainer(AgentTrainer):
    def __init__(self, name, policy, obs_shape, act_shape, args):
        self.name = name
        self.policy = policy  # NOTE: policy must be instantiated before
        assert isinstance(policy, torch.nn.Module), 'policy must be an instantiated instance of torch.nn.Module'
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        # training args
        self.args = args
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.batch_size = args['batch_size']
        if args['optimizer'] == 'adam':
            self.optimizer = optim.Adam(policy.parameters(), lr=self.lrate)
        else:
            self.optimizer = optim.RMSprop(policy.parameters(), lr=self.lrate)
        self.replay_buffer = ReplayBuffer(
                                args['replay_buffer_size'],  # replay buffer must be small!
                                args['frame_history_len'],
                                action_shape=[self.act_dim],
                                action_type=np.float32)
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.sample_counter = 0

    def action(self):
        frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
        raw_x = self._process_frames(frames, volatile=True)
        #raw_x = Variable(torch.from_numpy(frames.transpose([0, 3, 1, 2])), volatile=True)
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
        # Store transition in the replay buffer.
        full_act = np.concatenate(act).squeeze()
        self.replay_buffer.store_effect(idx, full_act, rew, (done or terminal), info)
        if done or terminal:
            self.sample_counter += 1

    def preupdate(self):
        pass

    def update(self):
        if self.sample_counter != self.args['batch_size']: return None
        self.sample_counter = 0

        obs, full_act, rew, _, done = self.replay_buffer.sample(-1)
        act = split_batched_array(full_act, self.act_shape)
        ret = np.stack(discount_with_dones(rew, done, self.gamma))
        ret_batch = Variable(torch.from_numpy((ret - np.mean(ret)) / np.std(ret)).type(FloatTensor), requires_grad=False)  # return

        # training
        #obs_batch = Variable(torch.from_numpy(obs.transpose([0, 3, 1, 2])).type(ByteTensor).type(FloatTensor)) / 255.0
        obs_batch = self._process_frames(obs)
        act_batch = [Variable(torch.from_numpy(a).type(FloatTensor), requires_grad=False) for a in act]
        self.policy(obs_batch)  # forward pass
        loss = (self.policy.logprob(act_batch) * ret_batch).mean()
        loss_val = loss.view(-1).data.cpu().numpy()[0]
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        ent = self.policy.entropy().mean()
        ent_val = ent.view(-1).data.cpu().numpy()[0]
        return dict(policy_loss=loss_val, policy_entropy=ent_val)
