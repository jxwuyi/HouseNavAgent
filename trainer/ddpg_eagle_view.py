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


def create_replay_buffer(action_shape, action_type, eagle_shape, args):
    partition = []
    default_partition = None
    if 'dist_sample' in args:
        n_partition = 20
        part_func = lambda info: min(int(info['scaled_dist'] * n_partition),n_partition-1)
        partition = [(n_partition, part_func)]
        default_partition = 0
    return FullReplayBuffer(
        args['replay_buffer_size'],
        args['frame_history_len'],
        action_shape=action_shape,
        action_type=action_type,
        partition=partition,
        default_partition=default_partition,
        extra_info_shapes=[eagle_shape, [4]],  # eagleMap, front_direction
        extra_info_types=[np.uint8, np.float32])


class EagleDDPGTrainer(DDPGTrainer):
    def __init__(self, name, policy_creator, critic_creator,
                 obs_shape, eagle_shape, act_shape, args, replay_buffer=None):
        if replay_buffer is None:
            replay_buffer = create_replay_buffer([sum(act_shape)], np.float32, eagle_shape, args)
        super(EagleDDPGTrainer, self).__init__(name, policy_creator, critic_creator,
                                          obs_shape, act_shape, args,
                                          replay_buffer=replay_buffer)

    def process_experience(self, idx, act, rew, done, terminal, info):
        # Store transition in the replay buffer.
        full_act = np.concatenate(act).squeeze()
        eagle_view = np.array(info['eagle_map'])
        front_dir = np.concatenate([info['front'], info['right']])
        self.replay_buffer.store_effect(idx, full_act, rew, (done or terminal), info, extra_infos=[eagle_view, front_dir])
        self.sample_counter += 1

    def update(self):
        if (self.sample_counter < self.args['update_freq']) or \
           not self.replay_buffer.can_sample(self.batch_size * self.args['episode_len']):
            return None
        self.sample_counter = 0
        self.train()
        tt = time.time()

        obs, full_act, rew, obs_next, done, extra_infos, extra_infos_next = \
            self.replay_buffer.sample(self.batch_size, collect_extras=True, collect_extra_next=True)
        eagle_maps, front_dir = extra_infos
        eagle_maps_next, front_dir_next = extra_infos_next
        #act = split_batched_array(full_act, self.act_shape)
        time_counter[-1] += time.time() - tt
        tt = time.time()

        # convert to variables
        obs_n = self._process_frames(obs)
        obs_next_n = self._process_frames(obs_next, volatile=True)
        full_act_n = Variable(torch.from_numpy(full_act)).type(FloatTensor)
        rew_n = Variable(torch.from_numpy(rew), volatile=True).type(FloatTensor)
        done_n = Variable(torch.from_numpy(done), volatile=True).type(FloatTensor)
        eagle_n = Variable(torch.from_numpy(eagle_maps)).type(ByteTensor).type(FloatTensor)
        eagle_next_n = Variable(torch.from_numpy(eagle_maps_next), volatile=True).type(ByteTensor).type(FloatTensor)
        front_n = Variable(torch.from_numpy(front_dir)).type(FloatTensor)
        front_next_n = Variable(torch.from_numpy(front_dir_next), volatile=True).type(FloatTensor)
        full_act_n = torch.cat([full_act_n, front_n], dim=-1)

        time_counter[0] += time.time() - tt
        tt = time.time()

        # train q network
        common.debugger.print('Grad Stats of Q Update ...', False)
        target_act_next = torch.cat(self.target_p(obs_next_n) + [front_next_n], dim=-1)
        target_q_next = self.target_q(eagle_next_n, target_act_next)  # use eagle_view
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile = False
        current_q = self.q(eagle_n, full_act_n)   # use eagle view
        q_norm = (current_q * current_q).mean().squeeze()  # l2 norm
        q_loss = F.smooth_l1_loss(current_q, target_q) + self.args['critic_penalty']*q_norm  # huber

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
        new_act_n = self.p(obs_n)  # NOTE: maybe use <gumbel_noise=None> ?
        new_act_n = torch.cat(new_act_n + [front_n], dim=-1)
        q_val = self.q(eagle_n, new_act_n)
        p_loss = -q_val.mean().squeeze()
        p_ent = self.p.entropy().mean().squeeze()
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

    def save(self, save_dir, version="", prefix="EagleDDPG"):
        super(EagleDDPGTrainer, self).save(save_dir, version, prefix)

    def load(self, save_dir, version="", prefix="EagleDDPG"):
        super(EagleDDPGTrainer, self).load(save_dir, version, prefix)
