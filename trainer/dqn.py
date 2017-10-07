from headers import *
import utils
from utils import *
from replay_buffer import *
from trainer.qac import *
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

class DQNTrainer(QACTrainer):
    def __init__(self, name, model_creator,
                 obs_shape, act_dim, args, replay_buffer=None):
        super(DQNTrainer, self).__init__(name, model_creator, obs_shape, act_dim, args, replay_buffer)
        # handle multi-task training
        self.multi_target = args['multi_target']
        self._target = 0
        if self.multi_target:
            self.target_buffer = np.zeros(args['replay_buffer_size'], dtype=np.uint8)

    def set_target(self, target):
        self._target = common.target_instruction_dict[target]

    def process_experience(self, idx, act, rew, done, terminal, info):
        super(DQNTrainer, self).process_experience(idx, act, rew, done, terminal, info)
        # handle multi-task learning
        if self.multi_target: self.target_buffer[idx] = common.target_instruction_dict[info['target_room']]

    def action(self, signal_level=None):
        self.eval()
        if (signal_level is not None) and (np.random.rand() > signal_level):
            # epsilon exploration, random policy
            actions = np.random.choice(self.act_dim)
        else:
            if self.multi_target:
                target = np.zeros([1, common.n_target_instructions], dtype=np.uint8)
                target[0, self._target] = 1
                target_n = Variable(torch.from_numpy(target).type(FloatTensor), volatile=True)
            else:
                target_n = None
            frames = self.replay_buffer.encode_recent_observation()[np.newaxis, ...]
            frames = self._process_frames(frames, volatile=True)
            q_value = self.net(frames, only_q_value=True, target=target_n)
            actions = torch.max(q_value, dim=1)[1]
            actions = actions.squeeze()  # [batch]
            if use_cuda:
                actions = actions.cpu()
            actions = actions.data.numpy()[0]
        #assert ((actions >= 0) and (actions < self.act_dim)), 'act_dim = {}, received action = {}'.format(self.act_dim, actions)
        return actions

    def update(self):
        if (self.sample_counter < self.args['update_freq']) or \
           not self.replay_buffer.can_sample(self.batch_size * min(self.args['update_freq'], 20)):
            return None
        self.sample_counter = 0
        self.train()
        tt = time.time()

        obs, act, rew, obs_next, done = \
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
        act_n = Variable(torch.from_numpy(act)).type(LongTensor)
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        done_n = Variable(torch.from_numpy(done), volatile=True).type(FloatTensor)
        if self.multi_target:
            target_n = Variable(torch.from_numpy(targets).type(FloatTensor))
        else:
            target_n = None

        time_counter[0] += time.time() - tt
        tt = time.time()

        # compute critic loss
        target_q_val_next = self.target_net(obs_next_n, only_q_value=True, target=target_n)
        # double Q learning
        target_act_next = torch.max(self.net(obs_next_n, only_q_value=True, target=target_n), dim=1, keepdim=True)[1]
        target_q_next = torch.gather(target_q_val_next, 1, target_act_next)
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile=False
        current_q_val = self.net(obs_n, only_q_value=True, target=target_n)
        current_q = torch.gather(current_q_val, 1, act_n.view(-1, 1))
        q_norm = (current_q * current_q).mean().squeeze()
        q_loss = F.smooth_l1_loss(current_q, target_q)

        common.debugger.print('>> Q_Loss = {}'.format(q_loss.data.mean()), False)
        common.debugger.print('>> Q_Norm = {}'.format(q_norm.data.mean()), False)

        total_loss = q_loss.mean()
        if self.args['critic_penalty'] > 1e-10:
            total_loss += self.args['critic_penalty']*q_norm

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

        return dict(critic_norm=q_norm.data.cpu().numpy()[0],
                    critic_loss=q_loss.data.cpu().numpy()[0])

    def train(self):
        self.net.train()
        #self.target_net.train()

    def eval(self):
        self.net.eval()

    def save(self, save_dir, version="", prefix="DQN"):
        super(DQNTrainer, self).save(save_dir, version, prefix)

    def load(self, save_dir, version="", prefix="DQN"):
        super(DQNTrainer, self).load(save_dir, version, prefix)
