from headers import *
import utils

import os, sys, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import policy
import trainer
from policy.simple_cnn_gumbel import CNNGumbelPolicy as Policy
from trainer.pg import PolicyGradientTrainer as PGTrainer
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
    assert (False)

frame_history_len = 3
resolution = (200, 150)
observation_shape = (3 * frame_history_len, resolution[0], resolution[1])
action_shape = (4, 2)
colide_res = 1000

def genCacheFile(houseID):
    return prefix + houseID + '/cachedmap1k.pkl'

#######################

def create_args(gamma = 0.9, lrate = 0.01, episode_len = 50, batch_size = 1024,
                replay_buffer_size = int(1e5),
                grad_clip = 2, optimizer = 'adam'):
    return dict(gamma=gamma, lrate=lrate, episode_len=episode_len,
                batch_size=batch_size, replay_buffer_size=replay_buffer_size,
                frame_history_len=frame_history_len, grad_clip=grad_clip,
                optimizer=optimizer)

def create_default_args(algo = 'pg'):
    if algo == 'pg':  # policy gradient
        return create_args(0.9, 0.01, 10, 100, 1000)
    elif algo == 'ddpg':  # ddpg
        return create_args(0.9, 0.01, 75, 1024, int(1e6))
    else: assert (False)

def create_policy(inp_shape, act_shape):
    # assume CNN Policy
    policy = Policy(inp_shape, act_shape,
                    hiddens=[32, 32, 16, 8],
                    kernel_sizes=5, strides=2,
                    activation = F.relu,  # F.elu
                    use_batch_norm = True)
    if use_cuda:
        policy.cuda()
    return policy

def create_trainer(policy, args):
    # self, name, policy, obs_shape, act_shape, args)
    trainer = PGTrainer('PolicyGradientTrainer',policy,
                        observation_shape, (4, 2), args)
    return trainer

def create_env(k = 0, linearReward=False):
    houseID = all_houseIDs[k]
    objFile = prefix + houseID + '/house.obj'
    jsonFile = prefix + houseID + '/house.json'
    cachedFile = genCacheFile(houseID)
    assert os.path.isfile(cachedFile), '[Warning] No Cached Map File Found for House <{}> (id = {})!'.format(houseID, k)
    world = World(jsonFile, objFile, csvFile, colide_res, CachedFile=cachedFile)
    env = HouseEnv(world, resolution=resolution, linearReward=linearReward)  # currently use indicator reward
    return env

def train(iters = 100000, report_rate = 400, save_rate = 2000, eval_range = 400,
         log_dir = './temp', save_dir = './_model_', algo='pg'):
    args = create_default_args(algo)
    policy = create_policy(observation_shape, action_shape)
    trainer = create_trainer(policy, args)
    env = create_env()
    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Training')

    episode_rewards = [0.0]

    obs = env.reset()
    obs = obs.transpose([1, 0, 2])
    print('Observation Shape = {}'.format(obs.shape))

    episode_step = 0
    t = 0
    best_res = -1e50
    elap = time.time()
    print('Starting iterations...')
    while(len(episode_rewards) <= iters):
        idx = trainer.process_observation(obs)
        # get action
        action = trainer.action()
        # environment step
        obs, rew, done, info = env.step(action)
        obs = obs.transpose([1, 0, 2])
        episode_step += 1
        terminal = (episode_step >= args['episode_len'])
        # collect experience
        trainer.process_experience(idx, action, rew, done, terminal)
        episode_rewards[-1] += rew

        if done or terminal:
            obs = env.reset()
            obs = obs.transpose([1, 0, 2])
            episode_step = 0
            episode_rewards.append(0)

        # update all trainers
        trainer.preupdate()
        loss, ent = trainer.update()

        # save results
        if (terminal and (len(episode_rewards) % save_rate == 0)) or \
            (len(episode_rewards) > iters):
            trainer.save(save_dir + '/' + trainer.name + '.pkl')
            logger.print('Successfully Saved to <{}>'.format(save_dir + '/' + trainer.name + '.pkl'))
            if np.mean(episode_rewards[-eval_range:]) > best_res:
                best_res = np.mean(episode_rewards[-eval_range:])
                trainer.save(save_dir + '/' + trainer.name + '_best.pkl')

        # display training output
        if (terminal and (len(episode_rewards) % report_rate == 0) and (algo != 'pg')) or \
            ((algo == 'pg') and (loss is not None)):
            logger.print('Episode#%d, Time Elapsed = %.3f min' % (len(episode_rewards), (time.time()-elap) / 60))
            logger.print('-> Total Samples: %d' % t)
            logger.print('  >> Loss    = %.4f' % loss)
            logger.print('  >> Entropy = %.4f' % ent)
            logger.print('  >> Reward  = %.4f' % np.mean(episode_rewards[-eval_range:]))

        t += 1


if __name__ == '__main__':
    train()
