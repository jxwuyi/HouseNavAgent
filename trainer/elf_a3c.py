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
import torch.autograd as autograd

class ELF_A3CTrainer(ELFTrainer):
    def __init__(self, name, model_creator,
                 obs_shape, act_dim, args):
        super(ELF_A3CTrainer, self).__init__()
        self.name = name
        self.net = model_creator()
        assert isinstance(self.net, torch.nn.Module), \
            'model must be an instantiated instance of torch.nn.Module'

        self.obs_shape = obs_shape
        self.act_dim = act_dim
        # training args
        self.args = args
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.batch_size = args['batch_size']
        if args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.net.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.max_episode_len = args['episode_len']
        self.grad_norm_clip = args['grad_clip']
        self.sample_counter = 0

    def train(self, cpu_batch, gpu_batch):
        pass

    def actor(self, cpu_batch, gpu_batch):
        pass
