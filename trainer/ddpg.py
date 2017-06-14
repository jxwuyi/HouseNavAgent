from headers import *
from utils import *
import common
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


import time


def make_update_exp(vals, target_vals, rate=1e-3):
    target_dict = target_vals.state_dict()
    val_dict = vals.state_dict()
    for k in target_dict.keys():
        target_dict[k] = target_dict[k] * (1 - rate) + rate * val_dict[k]
    target_vals.load_state_dict(target_dict)


class DDPGTrainer(AgentTrainer):
    def __init__(self, name, policy_creator, critic_creator,
                 obs_shape, act_shape, args):
        self.name = name
        self.p = policy_creator()
        self.q = critic_creator()
        assert isinstance(self.p, torch.nn.Module), \
            'actor must be an instantiated instance of torch.nn.Module'
        assert isinstance(self.q, torch.nn.Module), \
            'critic must be an instantiated instance of torch.nn.Module'
        self.target_p = policy_creator()
        self.target_q = critic_creator()
        self.target_p.load_state_dict(self.p.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        # training args
        self.args = args
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.critic_lrate = args['critic_lrate']
        self.batch_size = args['batch_size']
        if args['optimizer'] == 'adam':
            self.p_optim = optim.Adam(self.p.parameters(), lr=self.lrate)
            self.q_optim = optim.Adam(self.q.parameters(), lr=self.critic_lrate)
        else:
            self.p_optim = optim.RMSprop(self.p.parameters(), lr=self.lrate)
            self.q_optim = optim.RMSprop(self.q.parameters(), lr=self.critic_lrate)
        self.target_update_rate = args['target_net_update_rate'] or 1e-3
        self.replay_buffer = ReplayBuffer(
                                args['replay_buffer_size'],
                                args['frame_history_len'],
                                action_shape=[self.act_dim],
                                action_type=np.float32)
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.sample_counter = 0

    def action(self):
        frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
        batched_actions = self.p(self._process_frames(frames, volatile=True))
        if use_cuda:
            cpu_actions = [a.cpu() for a in batched_actions]
        else:
            cpu_actions = batched_actions
        return [a[0].data.numpy() for a in cpu_actions]

    def process_observation(self, obs):
        idx = self.replay_buffer.store_frame(obs)
        return idx

    def process_experience(self, idx, act, rew, done, terminal):
        # Store transition in the replay buffer.
        full_act = np.concatenate(act).squeeze()
        self.replay_buffer.store_effect(idx, full_act, rew, done or terminal)
        self.sample_counter += 1

    def preupdate(self):
        pass

    def update(self):
        if (self.sample_counter < self.args['update_freq']) or \
           not self.replay_buffer.can_sample(self.batch_size * self.args['episode_len']):
            return None
        self.sample_counter = 0


        tt = time.time()


        obs, full_act, rew, obs_next, done = \
            self.replay_buffer.sample(self.batch_size)
        #act = split_batched_array(full_act, self.act_shape)
        time_counter[-1] += time.time() - tt
        tt = time.time()

        # convert to variables
        obs_n = self._process_frames(obs)
        obs_next_n = self._process_frames(obs_next, volatile=True)
        full_act_n = Variable(torch.from_numpy(full_act)).type(FloatTensor)
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        done_n = Variable(torch.from_numpy(done), volatile=True).type(FloatTensor)

        time_counter[0] += time.time() - tt
        tt = time.time()

        # train q network
        target_act_next = self.target_p(obs_next_n)
        target_q_next = self.target_q(obs_next_n, target_act_next)
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile = False
        current_q = self.q(obs_n, full_act_n)
        q_norm = (current_q * current_q).mean().squeeze()  # l2 norm
        q_loss = F.smooth_l1_loss(current_q, target_q) + 0.001*q_norm  # huber
        self.q_optim.zero_grad()
        q_loss.backward()
        if self.grad_norm_clip is not None:
            nn.utils.clip_grad_norm(self.q.parameters(), self.grad_norm_clip)
        self.q_optim.step()

        # train p network
        new_act_n = self.p(obs_n)  # NOTE: maybe use <gumbel_noise=None> ?
        q_val = self.q(obs_n, new_act_n)
        p_loss = -q_val.mean().squeeze()
        p_ent = self.p.entropy().mean().squeeze()
        if self.args['ent_penalty'] is not None:
            p_loss -= self.args['ent_penalty'] * p_ent  # encourage exploration
        self.p_optim.zero_grad()
        self.q_optim.zero_grad()  # important!! clear the grad in Q
        p_loss.backward()
        if self.grad_norm_clip is not None:
            nn.utils.clip_grad_norm(self.p.parameters(), self.grad_norm_clip)
        self.p_optim.step()


        time_counter[1] += time.time() -tt
        tt =time.time()


        # update target networks
        make_update_exp(self.p, self.target_p, rate=self.target_update_rate)
        make_update_exp(self.q, self.target_q, rate=self.target_update_rate)


        time_counter[2] += time.time()-tt

        return dict(policy_loss=p_loss.data.cpu().numpy()[0],
                    policy_entropy=p_ent.data.cpu().numpy()[0],
                    critic_loss=q_loss.data.cpu().numpy()[0])

    def train(self):
        self.p.train()
        self.q.train()
        self.target_p.train()
        self.target_q.train()

    def eval(self):
        self.p.eval()

    def save(self, save_dir, version=""):
        if len(version) > 0:
            version = "_" + version
        if save_dir[-1] != '/':
            save_dir += '/'
        filename = save_dir + "DDPG_" + self.name + version + '.pkl'
        all_data = [self.p.state_dict(), self.target_p.state_dict(),
                    self.q.state_dict(), self.target_q.state_dict()]
        torch.save(all_data, filename)

    def load(self, save_dir, version=""):
        if os.path.isfile(save_dir) or (version is None):
            filename = save_dir
        else:
            if len(version) > 0:
                version = "_" + version
            if save_dir[-1] != '/':
                save_dir += '/'
            filename = save_dir + "DDPG_" + self.name + version + '.pkl'
        all_data = torch.load(filename)
        self.p.load_state_dict(all_data[0])
        self.target_p.load_state_dict(all_data[1])
        self.q.load_state_dict(all_data[2])
        self.target_q.load_state_dict(all_data[3])
