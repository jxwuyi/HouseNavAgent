from headers import *
import utils
from utils import *
from replay_buffer import *
import common
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd


import time


def make_update_exp(vals, target_vals, rate=1e-3):
    target_dict = target_vals.state_dict()
    val_dict = vals.state_dict()
    for k in target_dict.keys():
        target_dict[k] = target_dict[k] * (1 - rate) + rate * val_dict[k]
    target_vals.load_state_dict(target_dict)


def create_replay_buffer(args):
    if 'dist_sample' not in args:
        return ReplayBuffer(
           args['replay_buffer_size'],
           args['frame_history_len'])
    else:
        n_partition = 20
        part_func = lambda info: min(int(info['scaled_dist'] * n_partition),n_partition-1)
        return FullReplayBuffer(
            args['replay_buffer_size'],
            args['frame_history_len'],
            partition=[(n_partition, part_func)],
            default_partition=0)


class A2CTrainer(AgentTrainer):
    def __init__(self, name, model_creator,
                 obs_shape, act_dim, args, replay_buffer=None):
        self.name = name
        self.net = model_creator()
        assert isinstance(self.net, torch.nn.Module), \
            'model must be an instantiated instance of torch.nn.Module'
        self.target_net = model_creator()
        self.target_net.load_state_dict(self.net.state_dict())

        self.obs_shape = obs_shape
        self.act_dim = act_dim
        # training args
        self.args = args
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.critic_lrate = args['critic_lrate']
        self.batch_size = args['batch_size']
        if args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.target_update_rate = args['target_net_update_rate'] or 1e-3
        self.replay_buffer = replay_buffer or create_replay_buffer(args)
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.sample_counter = 0

    def action(self, signal_level = None):
        self.eval()
        if (signal_level is not None) and (np.random.rand() > signal_level):
            # epsilon exploration, random policy
            actions = np.random.choice(self.act_dim)
        else:
            frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
            frames = self._process_frames(frames, volatile=True)
            actions = self.net(frames, return_value=False).squeeze()
            if use_cuda:
                actions = actions.cpu()
            actions = actions.data.numpy()[0]
        return actions

    def process_observation(self, obs):
        idx = self.replay_buffer.store_frame(obs)
        return idx

    def process_experience(self, idx, act, rew, done, terminal, info):
        # Store transition in the replay buffer.
        self.replay_buffer.store_effect(idx, act, rew, (done or terminal), info)
        self.sample_counter += 1

    def preupdate(self):
        pass

    def update(self):
        if (self.sample_counter < self.args['update_freq']) or \
           not self.replay_buffer.can_sample(self.batch_size * min(self.args['update_freq'], 10)):
            return None
        self.sample_counter = 0
        self.train()
        tt = time.time()

        obs, act, rew, obs_next, done = \
            self.replay_buffer.sample(self.batch_size)
        #act = split_batched_array(full_act, self.act_shape)
        time_counter[-1] += time.time() - tt
        tt = time.time()

        # convert to variables
        obs_n = self._process_frames(obs)
        obs_next_n = self._process_frames(obs_next, volatile=True)
        act_n = torch.from_numpy(act).type(LongTensor)
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        done_n = Variable(torch.from_numpy(done), volatile=True).type(FloatTensor)

        time_counter[0] += time.time() - tt
        tt = time.time()

        # compute critic loss
        target_q_next = self.target_net(obs_next_n, only_value=True)
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile=False
        current_act, current_q = self.net(obs_n, return_value=True)
        q_norm = (current_q * current_q).mean().squeeze()
        q_loss = F.smooth_l1_loss(current_q, target_q)

        common.debugger.print('>> Q_Loss = {}'.format(q_loss.data.mean()), False)
        common.debugger.print('>> Q_Norm = {}'.format(q_norm.data.mean()), False)

        total_loss = q_loss.mean()
        if self.args['critic_penalty'] > 1e-10:
            total_loss += self.args['critic_penalty']*q_norm

        # compute policy loss
        # NOTE: currently 1-step lookahead!!! TODO: multiple-step lookahead
        raw_adv_ts = (rew_n - current_q).data
        #raw_adv_ts = (target_q - current_q).data   # use estimated advantage??
        adv_ts = (raw_adv_ts - raw_adv_ts.mean()) / (raw_adv_ts.std() + 1e-15)
        #current_act.reinforce(adv_ts)
        p_ent = self.net.entropy().mean()
        p_loss = self.net.logprob(act_n)
        p_loss = p_loss * Variable(adv_ts)
        p_loss = p_loss.mean()
        total_loss -= p_loss
        if self.args['ent_penalty'] is not None:
            total_loss -= self.args['ent_penalty'] * p_ent  # encourage exploration
        common.debugger.print('>> P_Loss = {}'.format(p_loss.data.mean()), False)
        common.debugger.print('>> P_Entropy = {}'.format(p_ent.data.mean()), False)

        # compute gradient
        self.optim.zero_grad()
        #autograd.backward([total_loss, current_act], [torch.ones(1), None])
        total_loss.backward()
        if self.grad_norm_clip is not None:
            #nn.utils.clip_grad_norm(self.q.parameters(), self.grad_norm_clip)
            utils.clip_grad_norm(self.net.parameters(), self.grad_norm_clip)
        self.optim.step()
        common.debugger.print('Stats of Model (*after* clip and opt)....', False)
        utils.log_parameter_stats(common.debugger, self.net)

        time_counter[1] += time.time() -tt
        tt =time.time()

        # update target networks
        make_update_exp(self.net, self.target_net, rate=self.target_update_rate)
        common.debugger.print('Stats of Target Network (After Update)....', False)
        utils.log_parameter_stats(common.debugger, self.target_net)

        time_counter[2] += time.time()-tt

        return dict(policy_loss=p_loss.data.cpu().numpy()[0],
                    policy_entropy=p_ent.data.cpu().numpy()[0],
                    critic_norm=q_norm.data.cpu().numpy()[0],
                    critic_loss=q_loss.data.cpu().numpy()[0])

    def train(self):
        self.net.train()
        #self.target_net.train()

    def eval(self):
        self.net.eval()

    def save(self, save_dir, version="", prefix="A2C"):
        if len(version) > 0:
            version = "_" + version
        if save_dir[-1] != '/':
            save_dir += '/'
        filename = save_dir + prefix + "_" + self.name + version + '.pkl'
        all_data = [self.net.state_dict(), self.target_net.state_dict()]
        torch.save(all_data, filename)

    def load(self, save_dir, version="", prefix="A2C"):
        if os.path.isfile(save_dir) or (version is None):
            filename = save_dir
        else:
            if len(version) > 0:
                version = "_" + version
            if save_dir[-1] != '/':
                save_dir += '/'
            filename = save_dir + prefix + "_" + self.name + version + '.pkl'
        all_data = torch.load(filename)
        self.net.load_state_dict(all_data[0])
        self.target_net.load_state_dict(all_data[1])
