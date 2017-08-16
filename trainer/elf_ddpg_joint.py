from trainer.elf_trainer_wrapper import *
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

class ELF_JointDDPGTrainer(ELFTrainer):
    def __init__(self, name, model_creator,
                 obs_shape, act_shape, args):
        super(ELF_JointDDPGTrainer, self).__init__(args['logger'], args['report_gap'], args['save_rate'], args['save_dir'])
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
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.batch_size = args['batch_size']
        if 'q_loss_coef' in args:
            self.q_loss_coef = args['q_loss_coef']
        else:
            self.q_loss_coef = default_q_loss_coef
        if args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.target_update_rate = args['target_net_update_rate'] or 1e-3
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.update_counter = 0
        self.sample_counter = 0

    def actor(self, cpu_batch, gpu_batch):
        #print(gpu_batch[0].keys())
        #print('[elf_ddpg] run actor! T = {}, batch shape = {}'.format(len(cpu_batch), gpu_batch[0]['s'].size()))
        self.eval()
        frames = Variable(gpu_batch[0]['s'], volatile=True)
        frames = frames.permute(0, 3, 1, 2)
        frames = (frames.type(FloatTensor) - 128.0) / 256.0
        batched_actions = self.net(frames, action=None, gumbel_noise=1.0, output_critic=False)  # assume gumbel_noise
        if isinstance(batched_actions, list):
            batched_actions = torch.cat(batched_actions, dim=-1)
        batched_actions = batched_actions.cpu()
        #print('[elf_ddpg] actor done!!! shape = {}'.format(batched_actions.size()), file=sys.stderr)
        return dict(a=batched_actions.data.numpy())

    def update(self, cpu_batch, gpu_batch):

        #print('[elf_ddpg] update!!!!')
        self.update_counter += 1
        self.train()
        tt = time.time()

        obs_n, obs_next_n, full_act_n, rew_n, done_n = self._process_elf_frames(gpu_batch, keep_time=False)  # collapse all the samples
        obs_n = (obs_n.type(FloatTensor) - 128.0) / 256.0
        obs_n = Variable(obs_n)
        obs_next_n = (obs_next_n.type(FloatTensor) - 128.0) / 256.0
        obs_next_n = Variable(obs_next_n, volatile=True)
        full_act_n = Variable(full_act_n)
        rew_n = Variable(rew_n, volatile=True)
        done_n = Variable(done_n, volatile=True)

        self.sample_counter += obs_n.size(0)

        time_counter[0] += time.time() - tt

        #print('[elf_ddpg] data loaded!!!!!')

        tt = time.time()

        self.optim.zero_grad()

        # train p network
        q_val = self.net(obs_n, action=None, output_critic=True)
        p_loss = -q_val.mean().squeeze()
        p_ent = self.net.entropy().mean().squeeze()
        if self.args['ent_penalty'] is not None:
            p_loss -= self.args['ent_penalty'] * p_ent  # encourage exploration
        common.debugger.print('>> P_Loss = {}'.format(p_loss.data.mean()), False)
        p_loss.backward()
        self.net.clear_critic_specific_grad()  # we do not need to compute q_grad for actor!!!

        # train q network
        common.debugger.print('Grad Stats of Q Update ...', False)
        target_q_next = self.target_net(obs_next_n, output_critic=True)
        target_q = rew_n + self.gamma * (1.0 - done_n) * target_q_next
        target_q.volatile = False
        current_q = self.net(obs_n, action=full_act_n, output_critic=True)
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

        stats = dict(policy_loss=p_loss.data.cpu().numpy()[0],
                     policy_entropy=p_ent.data.cpu().numpy()[0],
                     critic_norm=q_norm.data.cpu().numpy()[0],
                     critic_loss=q_loss.data.cpu().numpy()[0] / self.q_loss_coef,
                     eplen=cpu_batch[-1]['stats_eplen'].mean(),
                     avg_rew=cpu_batch[-1]['stats_rew'].mean())
        self.print_log(stats)

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
