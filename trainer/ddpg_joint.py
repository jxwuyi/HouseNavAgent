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


import time

default_q_loss_coef = 100.0

def make_update_exp(vals, target_vals, rate=1e-3):
    target_dict = target_vals.state_dict()
    val_dict = vals.state_dict()
    for k in target_dict.keys():
        target_dict[k] = target_dict[k] * (1 - rate) + rate * val_dict[k]
    target_vals.load_state_dict(target_dict)


def create_replay_buffer(action_shape, action_type, args):
    if 'dist_sample' not in args:
        return ReplayBuffer(
           args['replay_buffer_size'],
           args['frame_history_len'],
           action_shape=action_shape,
           action_type=action_type)
    else:
        n_partition = 20
        print('Use Distance Sampling, N_Partition = {}'.format(n_partition))
        part_func = lambda info: min(int(info['scaled_dist'] * n_partition),n_partition-1)
        return FullReplayBuffer(
            args['replay_buffer_size'],
            args['frame_history_len'],
            action_shape=action_shape,
            action_type=action_type,
            partition=[(n_partition, part_func)],
            default_partition=0)

class JointDDPGTrainer(AgentTrainer):
    def __init__(self, name, model_creator,
                 obs_shape, act_shape, args, replay_buffer=None):
        super(JointDDPGTrainer, self).__init__()
        self.name = name
        self.net = model_creator()
        assert isinstance(self.net, torch.nn.Module), \
            'joint-actor-critic must be an instantiated instance of torch.nn.Module'
        self.target_net = model_creator()
        self.target_net.load_state_dict(self.net.state_dict())

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        # training args
        self.args = args
        self.multi_target = args['multi_target']
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.critic_lrate = args['critic_lrate']
        self.batch_size = args['batch_size']
        self.start_train_samples = min(256, self.batch_size) * args['episode_len']
        #self.start_train_samples = 4 * self.batch_size # [test-use]
        if 'q_loss_coef' in args:
            self.q_loss_coef = args['q_loss_coef']
        else:
            self.q_loss_coef = default_q_loss_coef
        if args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.target_update_rate = args['target_net_update_rate'] or 1e-3
        self.replay_buffer = replay_buffer or create_replay_buffer([self.act_dim], np.float32, args)
        if self.multi_target:
            self.target_buffer = np.zeros(args['reply_buffer_size'], dtype=np.uint8)
            self._target = 0
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.sample_counter = 0

    def action(self, gumbel_noise=None):
        self.eval()
        frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
        frames = self._process_frames(frames, volatile=True)
        if self.multi_target:
            target = np.zeros([1, common.n_target_instructions], dtype=np.uint8)
            target[0, self._target] = 1
            target_n = Variable(torch.from_numpy(target).type(FloatTensor), volatile=True)
        else:
            target_n = None
        if gumbel_noise is not None:
            batched_actions = self.net(frames, action=None, gumbel_noise=gumbel_noise, output_critic=False, target=target_n)
        else:
            batched_actions = self.net(frames, action=None, gumbel_noise=None, output_critic=False, target=target_n)
        if use_cuda:
            cpu_actions = [a.cpu() for a in batched_actions]
        else:
            cpu_actions = batched_actions
        return [a[0].data.numpy() for a in cpu_actions]

    def process_observation(self, obs):
        idx = self.replay_buffer.store_frame(obs)
        return idx

    def set_target(self, target):
        self._target = common.target_instruction_dict[target]

    def process_experience(self, idx, act, rew, done, terminal, info):
        # Store transition in the replay buffer.
        full_act = np.concatenate(act).squeeze()
        self.replay_buffer.store_effect(idx, full_act, rew, (done or terminal), info)
        self.sample_counter += 1
        if self.multi_target:
            self.target_buffer[idx] = common.target_instruction_dict[info['target_room']]

    def preupdate(self):
        pass

    def update(self):
        if (self.sample_counter < self.args['update_freq']) or \
           not self.replay_buffer.can_sample(self.start_train_samples):
            return None
        self.sample_counter = 0
        self.train()
        tt = time.time()

        obs, full_act, rew, obs_next, done = \
            self.replay_buffer.sample(self.batch_size)
        if self.multi_target:
            target_idx = self.target_buffer[self.replay_buffer._idxes]
            targets = np.zeros((self.batch_size, common.n_target_instructions), dtype=np.uint8)
            targets[list(range(self.batch_size)), target_idx] = 1
        #act = split_batched_array(full_act, self.act_shape)
        time_counter[-1] += time.time() - tt
        tt = time.time()

        # convert to variables
        obs_n = self._process_frames(obs)
        obs_next_n = self._process_frames(obs_next, volatile=True)
        full_act_n = Variable(torch.from_numpy(full_act)).type(FloatTensor)
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        done_n = Variable(torch.from_numpy(done), volatile=True).type(FloatTensor)
        if self.multi_target:
            target_n = Variable(torch.from_numpy(targets).type(FloatTensor))
        else:
            target_n = None

        time_counter[0] += time.time() - tt
        tt = time.time()

        self.optim.zero_grad()

        # train p network
        q_val = self.net(obs_n, action=None, output_critic=True, target=target_n)
        p_loss = -q_val.mean().squeeze()
        p_ent = self.net.entropy().mean().squeeze()
        if self.args['ent_penalty'] is not None:
            p_loss -= self.args['ent_penalty'] * p_ent  # encourage exploration
        common.debugger.print('>> P_Loss = {}'.format(p_loss.data.mean()), False)
        p_loss.backward()
        self.net.clear_critic_specific_grad()  # we do not need to compute q_grad for actor!!!

        # train q network
        common.debugger.print('Grad Stats of Q Update ...', False)
        target_q_next = self.target_net(obs_next_n, output_critic=True, target=target_n)
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile = False
        current_q = self.net(obs_n, action=full_act_n, output_critic=True, target=target_n)
        q_norm = (current_q * current_q).mean().squeeze()  # l2 norm
        q_loss = F.smooth_l1_loss(current_q, target_q) + self.args['critic_penalty']*q_norm  # huber
        common.debugger.print('>> Q_Loss = {}'.format(q_loss.data.mean()), False)
        q_loss = q_loss * self.q_loss_coef
        q_loss.backward()

        # total_loss = q_loss + p_loss
        # grad clip
        if self.grad_norm_clip is not None:
            utils.clip_grad_norm(self.net.parameters(), self.grad_norm_clip)
        self.optim.step()

        common.debugger.print('Stats of P Network (after clip and opt)....', False)
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
                    critic_loss=q_loss.data.cpu().numpy()[0] / self.q_loss_coef)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def save(self, save_dir, version="", prefix="JointDDPG"):
        if len(version) > 0:
            version = "_" + version
        if save_dir[-1] != '/':
            save_dir += '/'
        filename = save_dir + prefix + "_" + self.name + version + '.pkl'
        all_data = [self.net.state_dict(), self.target_net.state_dict()]
        torch.save(all_data, filename)

    def load(self, save_dir, version="", prefix="JointDDPG"):
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
