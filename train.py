from headers import *
import common
import utils

import os, sys, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(iters = 100000, report_rate = 400, save_rate = 2000, eval_range = 400,
         log_dir = './temp', save_dir = './_model_', algo='pg', model_name = 'cnn'):
    args = common.create_default_args(algo)
    trainer = common.create_trainer(algo, model_name, args)
    env = common.create_env()
    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Training')

    episode_rewards = [0.0]

    obs = env.reset()
    obs = obs.transpose([1, 0, 2])
    logger.print('Observation Shape = {}'.format(obs.shape))

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
