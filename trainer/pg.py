from headers import *
from utils import *
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
        self.bacth_size = args['batch_size']
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
        raw_x = Variable(torch.from_numpy(frames.transpose([0, 3, 1, 2])), volatile=True).type(ByteTensor)
        _,_,batched_actions = self.policy(raw_x.type(FloatTensor))
        cpu_actions = [a.cpu() for a in batched_actions]
        return [a[0].data.numpy() for a in cpu_actions]

    def process_observation(self, obs):
        idx = self.replay_buffer.store_frame(obs)
        return idx

    def process_experience(self, idx, act, rew, done, terminal):
        # Store transition in the replay buffer.
        full_act = np.concatenate(act).squeeze()
        self.replay_buffer.store_effect(idx, full_act, rew, done or terminal)
        if done or terminal:
            self.sample_counter += 1

    def preupdate(self):
        pass

    def update(self):
        if self.sample_counter != self.args['batch_size']: return None, None
        self.sample_counter = 0

        obs, full_act, rew, _, done = self.replay_buffer.sample(-1)
        act = split_batched_array(full_act, self.act_shape)
        ret = np.stack(discount_with_dones(rew, done, self.gamma))
        ret = (ret - np.mean(ret)) / np.std(ret)  # return

        # training
        obs_batch = Variable(torch.from_numpy(obs.transpose([0, 3, 1, 2])).type(ByteTensor).type(FloatTensor), volatile=True) / 255.0
        act_batch = [Variable(torch.from_numpy(a).type(FloatTensor), volatile=True) for a in act]
        policy(obs)  # forward pass
        loss = (policy.logprob(act_batch) * ret).mean()
        loss_val = loss.view(-1).numpy()[0]
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm(policy.prameters(), args.grad_norm_clip)
        optimizer.step()
        return loss_val, policy.entropy().mean().view(-1).numpy()[0]
