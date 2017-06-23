from headers import *
import utils
from utils import *
from replay_buffer import *
from trainer.ddpg import DDPGTrainer
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


class RDPGTrainer(DDPGTrainer):
    def __init__(self, name, policy_creator, critic_creator,
                 obs_shape, act_shape, args, replay_buffer=None):
        if replay_buffer is None:
            replay_buffer = RNNReplayBuffer(
                                args['replay_buffer_size'],
                                args['episode_len'],  # max_seq_len
                                action_shape=[sum(act_shape)],
                                action_type=np.float32)
        super(RDPGTrainer, self).__init__(name, policy_creator, critic_creator,
                                          obs_shape, act_shape, args,
                                          replay_buffer=replay_buffer)

        self.h = None  # last hidden
        self.a = None  # last action
        # training args specialized for reccurent nets
        self.batch_len = args['batch_len']

    def reset_agent(self):
        self.h = self.p._get_zero_state(1)  # batch size = 1
        #self.h.volatile = True
        self.a = None

    def action(self, gumbel_noise=True):
        self.eval()
        frames = self.replay_buffer.encode_recent_observation()[np.newaxis, np.newaxis, ...]  # [batch=1, seq_len=1]
        frames = self._process_frames(frames, volatile=True, merge_dim=False)
        if gumbel_noise:
            batched_actions, new_h = self.p(frames, h=self.h, act=self.a)
        else:
            batched_actions, new_h = self.p(frames, h=self.h, act=self.a, gumbel_noise=None)
        self.h = new_h
        self.a = batched_actions
        if use_cuda:
            cpu_actions = [a.squeeze().cpu() for a in batched_actions]
        else:
            cpu_actions = batched_actions.squeeze()
        return [a.data.numpy() for a in cpu_actions]

    def update(self):
        if (self.a is not None) or \
           not self.replay_buffer.can_sample(self.batch_size * 4):
            return None
        self.sample_counter = 0
        self.train()
        tt = time.time()

        obs, full_act, rew, msk, done, total_length = \
            self.replay_buffer.sample(self.batch_size)
        total_length = float(total_length)
        #act = split_batched_array(full_act, self.act_shape)
        time_counter[-1] += time.time() - tt
        tt = time.time()


        # convert to variables
        _full_obs_n = self._process_frames(obs, merge_dim=False, return_variable=False)  # [batch, seq_len+1, ...]
        batch = _full_obs_n.size(0)
        seq_len = _full_obs_n.size(1) - 1
        n_samples = batch * seq_len
        full_obs_n = Variable(_full_obs_n, volatile=True)
        obs_n = Variable(_full_obs_n[:, :-1, ...]).contiguous() # [batch, seq_len, ...]
        obs_next_n = Variable(_full_obs_n[:, 1:, ...], volatile=True).contiguous()
        img_c, img_h, img_w = obs_n.size(-3), obs_n.size(-2), obs_n.size(-1)
        packed_obs_n = obs_n.view(-1, img_c, img_h, img_w)
        packed_obs_next_n = obs_next_n.view(-1, img_c, img_h, img_w)
        full_act_n = Variable(torch.from_numpy(full_act)).type(FloatTensor)  # [batch, seq_len, ...]
        act_padding = Variable(torch.zeros(self.batch_size, 1, full_act_n.size(-1))).type(FloatTensor)
        pad_act_n = torch.cat([act_padding, full_act_n], dim=1)  # [batch, seq_len+1, ...]
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        msk_n = Variable(torch.from_numpy(msk)).type(FloatTensor)  # [batch, seq_len]
        done_n = Variable(torch.from_numpy(done)).type(FloatTensor)  # [batch, seq_len]



        time_counter[0] += time.time() - tt
        tt = time.time()

        # train q network
        common.debugger.print('Grad Stats of Q Update ...', False)

        full_target_act, _ = self.target_p(full_obs_n, act=pad_act_n)  # list([batch, seq_len+1, act_dim])
        target_act_next = torch.cat(full_target_act, dim=-1)[:, 1:, :]
        act_dim = target_act_next.size(-1)
        target_act_next = target_act_next.resize(batch * seq_len, act_dim)

        target_q_next = self.target_q(packed_obs_next_n, act=target_act_next)  #[batch * seq_len]
        target_q_next.view(batch, seq_len)
        target_q = (rew_n + self.gamma * done_n * target_q_next) * msk_n
        target_q = target_q.view(-1)
        target_q.volatile = False

        current_q = self.q(packed_obs_n, act=full_act_n.view(-1, act_dim)) * msk_n.view(-1)
        q_norm = (current_q * current_q).sum() / total_length  # l2 norm
        q_loss = F.smooth_l1_loss(current_q, target_q, size_average=False) / total_length \
                 + self.args['critic_penalty']*q_norm  # huber

        common.debugger.print('>> Q_Loss = {}'.format(q_loss.data.mean()), False)

        self.q_optim.zero_grad()
        q_loss.backward()

        common.debugger.print('Stats of Q Network (*before* clip and opt)....', False)
        utils.log_parameter_stats(common.debugger, self.q)

        if self.grad_norm_clip is not None:
            #nn.utils.clip_grad_norm(self.q.parameters(), self.grad_norm_clip)
            utils.clip_grad_norm(self.q.parameters(), self.grad_norm_clip)
        self.q_optim.step()

        # train p network
        new_act_n, _ = self.p(obs_n, act=pad_act_n[:, :-1, :])  # [batch, seq_len, act_dim]
        new_act_n = torch.cat(new_act_n, dim=-1)
        new_act_n = new_act_n.view(-1, act_dim)
        q_val = self.q(packed_obs_n, new_act_n) * msk_n.view(-1)
        p_loss = -q_val.sum() / total_length
        p_ent = self.p.entropy(weight=msk_n).sum() / total_length
        if self.args['ent_penalty'] is not None:
            p_loss -= self.args['ent_penalty'] * p_ent  # encourage exploration

        common.debugger.print('>> P_Loss = {}'.format(p_loss.data.mean()), False)

        self.p_optim.zero_grad()
        self.q_optim.zero_grad()  # important!! clear the grad in Q
        p_loss.backward()

        if self.grad_norm_clip is not None:
            #nn.utils.clip_grad_norm(self.p.parameters(), self.grad_norm_clip)
            utils.clip_grad_norm(self.p.parameters(), self.grad_norm_clip)
        self.p_optim.step()

        common.debugger.print('Stats of Q Network (in the phase of P-Update)....', False)
        utils.log_parameter_stats(common.debugger, self.q)
        common.debugger.print('Stats of P Network (after clip and opt)....', False)
        utils.log_parameter_stats(common.debugger, self.p)


        time_counter[1] += time.time() -tt
        tt =time.time()


        # update target networks
        make_update_exp(self.p, self.target_p, rate=self.target_update_rate)
        make_update_exp(self.q, self.target_q, rate=self.target_update_rate)

        common.debugger.print('Stats of Q Target Network (After Update)....', False)
        utils.log_parameter_stats(common.debugger, self.target_q)
        common.debugger.print('Stats of P Target Network (After Update)....', False)
        utils.log_parameter_stats(common.debugger, self.target_p)


        time_counter[2] += time.time()-tt

        return dict(policy_loss=p_loss.data.cpu().numpy()[0],
                    policy_entropy=p_ent.data.cpu().numpy()[0],
                    critic_norm=q_norm.data.cpu().numpy()[0],
                    critic_loss=q_loss.data.cpu().numpy()[0])

    def save(self, save_dir, version="", prefix="RDPG"):
        super(RDPGTrainer, self).save(save_dir, version, prefix)

    def load(self, save_dir, version="", prefix="RDPG"):
        super(RDPGTrainer, self).load(save_dir, version, prefix)
