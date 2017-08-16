import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from headers import *

if "Apple" in sys.version:
    # own mac PC
    path_to_elf_repo = '/Users/yiw/workroom/ELF'
elif "Red Hat" in sys.version:
    path_to_elf_repo = '/home/yiw/code/ELF'
else:
    assert(False)
sys.path.insert(0, path_to_elf_repo)

default_report_gap = 50

def get_key(h, key, return_key = False):
    if key not in h:
        key = 'last_' + key
        if key not in h:
            if return_key:
                return None, ''
            else:
                return None
    if return_key:
        return h[key], key
    else:
        return h[key]


class ELFTrainer(AgentTrainer):
    def __init__(self, logger, report_gap=default_report_gap, save_rate=1000, save_dir='./log'):
        super(ELFTrainer, self).__init__()
        self.logger = logger
        self.report_gap = report_gap
        self.save_rate = save_rate
        self.save_dir = save_dir
        self.start_time = time.time()
        self.best_result = -1e50

    def _process_elf_frames(self, data_batch, keep_time=False, proc_next=True):
        """
        either cpu_batch or gpu_batch
        <state> key = "s"
        data_batch[0...T]['s']=ndarray([n_games, n_row, n_col, n_channel])
          >> frame stacking was performed in the environment
          >> we skip the last time step (only keep it in the next_batch)
        when keep_time==True:
            return ndarray([T, n_game, n_channel, n_row, n_col])
        else
            return ndarray([T * n_games, n_channel, n_row, n_col])
        <reward> key = "r"
        <action> key = "a"
        <done> key = "done"
          >> here only consider terminal (done or out-of-max-steps)
        """
        all_sample = torch.stack([sample['s'] for sample in data_batch])
        all_sample = all_sample.permute(0, 1, 4, 2, 3)
        flag, next_key = get_key(data_batch[0], 'next', return_key=True)
        if flag is not None:
            T = len(data_batch)
            cur_sample = all_sample
            for sample in data_batch:
                assert not isinstance(sample[next_key], np.ndarray), 'Error!!! sample[{}] type = {}'.format(next_key, type(sample[next_key]))
            nxt_sample = torch.stack(list([sample[next_key] for sample in data_batch]))
            nxt_sample = nxt_sample.permute(0, 1, 4, 2, 3)
        else:
            T = len(data_batch) - 1   # drop the last frame
            cur_sample = all_sample[:-1]
            nxt_sample = all_sample[1:]
        assert((all_sample.type(FloatTensor) - nxt_sample.type(FloatTensor)).abs().max() > 1e-3), 'nxt_sample and cur_sample is the same!!!'
        actions = torch.stack(list([get_key(data_batch[t], 'a') for t in range(T)])) # drop last action
        rewards = torch.stack(list([get_key(data_batch[t], 'r') for t in range(T)]))
        dones = torch.stack(list([get_key(data_batch[t], 'done') for t in range(T)]))
        if not keep_time:
            n_chn, n_row, n_col = all_sample.size(2), all_sample.size(3), all_sample.size(4)
            cur_sample = cur_sample.contiguous().view(-1, n_chn, n_row, n_col)
            nxt_sample = nxt_sample.contiguous().view(-1, n_chn, n_row, n_col)
            n_act = actions.size(-1)
            actions = actions.view(-1, n_act)
            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
        return cur_sample, nxt_sample, actions, rewards, dones

    def action(self, obs):
        cpu_s = torch.from_numpy(obs).type(torch.ByteTensor)
        cpu_s = cpu_s.permute(2, 0, 1)
        n_chn, n_row, n_col = cpu_s.size()
        cpu_s = cpu_s.view(1, n_chn, n_row, n_col)
        gpu_s = cpu_s.type(ByteTensor).type(FloatTensor)
        cpu_batch = [dict(s=torch.from_numpy(cpu_s))]
        gpu_batch = [dict(s=torch.from_numpy(gpu_s))]
        ret = self.actor(cpu_batch, gpu_batch)
        act = ret['a'][0]
        if isinstance(act, torch.Tensor):
            act = act.numpy()
        return act

    def print_log(self, info):
        if self.update_counter % self.report_gap == 0:
            avg_rew = info['avg_rew']
            if not isinstance(avg_rew, float):
                avg_rew = avg_rew.mean()
            avg_eplen = info['eplen']
            if not isinstance(avg_eplen, float):
                avg_eplen = avg_eplen.mean()
            self.logger.print('Updates=%d, Time Elapsed = %.3f min' % (self.update_counter, (time.time()-self.start_time) / 60))
            self.logger.print('-> Total Samples: %d' % self.sample_counter)
            self.logger.print('-> Avg Episode Length: %.4f' % (avg_eplen))
            self.logger.print('-> Avg Reward: %.4f' % (avg_rew))
            if info is not None:
                for k in info:
                    self.logger.print('  >> %s = %.4f' % (k, info[k]))
            print('----> Data Loading Time = %.4f min' % (time_counter[0] / 60))
            print('----> Training Time = %.4f min' % (time_counter[1] / 60))
            print('----> Target Net Update Time = %.4f min' % (time_counter[2] / 60))

            if (self.update_counter % self.save_rate == 0):
                self.save(self.save_dir)
                self.logger.print('Successfully Saved to <{}>'.format(self.save_dir + '/' + self.name + '.pkl'))

            if avg_rew > self.best_result:
                self.best_result = avg_rew
                self.save(self.save_dir, 'best')
                self.logger.print('Best Result (rew={}) Successfully Saved!'.format(avg_rew))

    def update(self, cpu_batch, gpu_batch):
        raise NotImplementedError()

    def actor(self, cpu_batch, gpu_batch):
        raise NotImplementedError()
