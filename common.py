from headers import *

import sys, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.simple_cnn_gumbel import CNNGumbelPolicy as CNNPolicy
from policy.vanila_random_policy import VanilaRandomPolicy as RandomPolicy
from policy.ddpg_cnn_critic import DDPGCNNCritic as DDPGCritic
from trainer.pg import PolicyGradientTrainer as PGTrainer
from trainer.nop import NOPTrainer
from trainer.ddpg import DDPGTrainer
from environment import SimpleHouseEnv as HouseEnv
from world import World

all_houseIDs = ['00065ecbdd7300d35ef4328ffe871505',
'cf57359cd8603c3d9149445fb4040d90', '31966fdc9f9c87862989fae8ae906295', 'ff32675f2527275171555259b4a1b3c3',
'7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86', '775941abe94306edc1b5820e3a992d75',
'32e53679b33adfcc5a5660b8c758cc96', '4383029c98c14177640267bd34ad2f3c', '0884337c703e7c25949d3a237101f060',
'492c5839f8a534a673c92912aedc7b63', 'a7e248efcdb6040c92ac0cdc3b2351a6', '2364b7dcc432c6d6dcc59dba617b5f4b',
'e3ae3f7b32cf99b29d3c8681ec3be321', 'f10ce4008da194626f38f937fb9c1a03', 'e6f24af5f87558d31db17b86fe269cf2',
'1dba3a1039c6ec1a3c141a1cb0ad0757', 'b814705bc93d428507a516b866efda28', '26e33980e4b4345587d6278460746ec4',
'5f3f959c7b3e6f091898caa8e828f110', 'b5bd72478fce2a2dbd1beb1baca48abd', '9be4c7bee6c0ba81936ab0e757ab3d61']

if "Apple" in sys.version:
    # own mac laptop
    prefix = '/Users/yiw/Downloads/data/house/'
    csvFile = '/Users/yiw/Downloads/data/metadata/ModelCategoryMapping.csv'
elif "Red Hat" in sys.version:
    # dev server
    prefix = '/home/yiw/local/data/houses-yiwu/'
    csvFile = '/home/yiw/local/data/houses-yiwu/ModelCategoryMapping.csv'
else:
    # fair server
    assert False, 'Unable to locate data folder..... Please edit <common.py>'

frame_history_len = 4
#resolution = (200, 150)
resolution = (120, 90)
observation_shape = (3 * frame_history_len, resolution[0], resolution[1])
action_shape = (4, 2)
colide_res = 1000


def genCacheFile(houseID):
    return prefix + houseID + '/cachedmap1k.pkl'

#######################


def create_args(gamma = 0.9, lrate = 0.01, episode_len = 50, batch_size = 1024,
                replay_buffer_size = int(1e5),
                grad_clip = 5, optimizer = 'adam',
                update_freq = 100, ent_penalty=None,
                target_net_update_rate = None):
    return dict(gamma=gamma, lrate=lrate, hardness = None,
                episode_len=episode_len,
                batch_size=batch_size, replay_buffer_size=replay_buffer_size,
                frame_history_len=frame_history_len,
                grad_clip=grad_clip,
                optimizer=optimizer,
                update_freq=update_freq,
                ent_penalty=None,
                target_net_update_rate=target_net_update_rate)


def create_default_args(algo='pg', gamma=None,
                        lrate=None, episode_len=None,
                        batch_size=None, update_freq=None):
    if algo == 'pg':  # policy gradient
        return create_args(gamma or 0.9, lrate or 0.01,
                           episode_len or 10, batch_size or 100, 1000)
    elif algo == 'ddpg':  # ddpg
        return create_args(gamma or 0.9, lrate or 0.001, episode_len or 50,
                           batch_size or 256, int(5e5),
                           update_freq=(update_freq or 100), ent_penalty=1e-3)
    elif algo == 'nop':
        return create_args()
    else:
        assert (False)


def create_policy(inp_shape, act_shape, name='cnn'):
    if name == 'random':
        policy = RandomPolicy(act_shape)
    elif name == 'cnn':
        # assume CNN Policy
        policy = CNNPolicy(inp_shape, act_shape,
                        hiddens=[32, 32, 16, 8],
                        kernel_sizes=5, strides=2,
                        activation = F.elu,  # F.relu
                        use_batch_norm = True)  # False
    else:
        assert False, 'Policy Undefined for <{}>'.format(name)
    if use_cuda:
        policy.cuda()
    return policy


def create_critic(inp_shape, act_shape, algo):
    act_dim = act_shape if isinstance(act_shape, int) else sum(act_shape)
    if algo == 'ddpg':
        critic = DDPGCritic(inp_shape, act_dim,
                            conv_hiddens=[32,16,8,4],
                            linear_hiddens=[64],
                            activation=F.relu)  # F.elu
    else:
        assert False, 'No critic defined for algo<{}>'.format(algo)
    if use_cuda:
        critic.cuda()
    return critic


def create_trainer(algo, model, args):
    # self, name, policy, obs_shape, act_shape, args)
    if algo == 'pg':
        policy = create_policy(observation_shape, action_shape, name=model)
        trainer = PGTrainer('PolicyGradientTrainer', policy,
                            observation_shape, action_shape, args)
    elif algo == 'nop':
        policy = create_policy(observation_shape, action_shape, name=model)
        trainer = NOPTrainer('NOPTrainer', policy, observation_shape, action_shape, args)
    elif algo == 'ddpg':
        assert(model == 'cnn')
        critic_gen = lambda: create_critic(observation_shape, action_shape, 'ddpg')
        policy_gen = lambda: create_policy(observation_shape, action_shape, 'cnn')
        trainer = DDPGTrainer('DDPGTrainer', policy_gen, critic_gen,
                              observation_shape, action_shape, args)
    else:
        assert False, 'Trainer not defined for <{}>'.format(algo)
    return trainer


def create_env(k=0, linearReward=False, hardness=None):
    houseID = all_houseIDs[k]
    objFile = prefix + houseID + '/house.obj'
    jsonFile = prefix + houseID + '/house.json'
    cachedFile = genCacheFile(houseID)
    assert os.path.isfile(cachedFile), '[Warning] No Cached Map File Found for House <{}> (id = {})!'.format(houseID, k)
    world = World(jsonFile, objFile, csvFile, colide_res, CachedFile=cachedFile)
    env = HouseEnv(world, resolution=resolution, linearReward=linearReward, hardness=hardness)
    return env
