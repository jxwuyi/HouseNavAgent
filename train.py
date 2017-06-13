from headers import *
import common
import utils

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(args=None,
          houseID=0, linearReward=False, algo='pg', model_name='cnn',
          iters=2000000, report_rate=20, save_rate=1000, eval_range=200,
          log_dir='./temp', save_dir='./_model_', warmstart=None):
    if args is None:
        args = common.create_default_args(algo)

    hardness = args['hardness']
    if hardness is not None:
        print('>>> Hardness Level = {}'.format(hardness))

    trainer = common.create_trainer(algo, model_name, args)
    env = common.create_env(houseID, linearReward, hardness)
    logger = utils.MyLogger(log_dir, True)

    if warmstart is not None:
        if os.path.exists(warmstart):
            logger.print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            logger.print('Warmstarting from save_dir <{}> with version <{}> ...'.format(save_dir, warmstart))
            trainer.load(save_dir, warmstart)

    logger.print('Start Training')

    episode_rewards = [0.0]

    obs = env.reset()
    obs = obs.transpose([1, 0, 2])
    logger.print('Observation Shape = {}'.format(obs.shape))

    episode_step = 0
    t = 0
    best_res = -1e50
    elap = time.time()
    update_times = 0
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
        if loss is not None:
            update_times += 1

        # save results
        if ((done or terminal) and (len(episode_rewards) % save_rate == 0)) or\
           (len(episode_rewards) > iters):
            trainer.save(save_dir)
            logger.print('Successfully Saved to <{}>'.format(save_dir + '/' + trainer.name + '.pkl'))
            if np.mean(episode_rewards[-eval_range:]) > best_res:
                best_res = np.mean(episode_rewards[-eval_range:])
                trainer.save(save_dir, "best")

        # display training output
        if ((update_times % report_rate == 0) and (algo != 'pg') and (loss is not None)) or \
            ((update_times == 0) and (algo != 'pg') and (len(episode_rewards) % 100 == 0) and (done or terminal)) or \
            ((algo == 'pg') and (loss is not None)):
            logger.print('Episode#%d, Updates=%d, Time Elapsed = %.3f min' % (len(episode_rewards), update_times, (time.time()-elap) / 60))
            logger.print('-> Total Samples: %d' % t)
            logger.print('-> Avg Episode Length: %.4f' % (t / len(episode_rewards)))
            if loss is not None:
                logger.print('  >> Loss    = %.4f' % loss)
                logger.print('  >> Entropy = %.4f' % ent)
            logger.print('  >> Reward  = %.4f' % np.mean(episode_rewards[-eval_range:]))
            print('----> Data Loading Time = %.4f min' % (time_counter[-1] / 60))
            print('----> GPU Data Transfer Time = %.4f min' % (time_counter[0] / 60))
            print('----> Training Time = %.4f min' % (time_counter[1] / 60))
            print('----> Target Net Update Time = %.4f min' % (time_counter[2] / 60))

        t += 1


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for 3D House Navigation")
    # Environment
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--linear-reward", action='store_true', default=False,
                        help="whether to use reward according to distance; o.w. indicator reward")
    # Core training parameters
    parser.add_argument("--algo", choices=['ddpg','pg'], default="ddpg", help="algorithm")
    parser.add_argument("--lrate", type=float, help="learning rate")
    parser.add_argument("--gamma", type=float, help="discount")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--max-episode-len", type=int, help="maximum episode length")
    parser.add_argument("--update-freq", type=int, help="update model parameters once every this many samples collected")
    parser.add_argument("--max-iters", type=int, default=int(2e6), help="maximum number of training episodes")
    parser.add_argument("--target-net-update-rate", type=float, help="update rate for target networks")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--report-rate", type=int, default=50, help="report training stats once every time this many training steps are performed")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = parse_args()
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)

    if cmd_args.linear_reward:
        print('Using Linear Reward Function in the Env!')

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    args = common.create_default_args(cmd_args.algo, cmd_args.gamma, cmd_args.lrate,
                               cmd_args.max_episode_len, cmd_args.batch_size,
                               cmd_args.update_freq)

    if cmd_args.target_net_update_rate is not None:
        args['target_net_update_rate']=cmd_args.target_net_update_rate

    if cmd_args.hardness is not None:
        args['hardness'] = cmd_args.hardness

    train(args,
          houseID=cmd_args.house, linearReward=cmd_args.linear_reward,
          algo=cmd_args.algo, iters=cmd_args.max_iters,
          report_rate=cmd_args.report_rate, save_rate=cmd_args.save_rate,
          log_dir=cmd_args.log_dir, save_dir=cmd_args.save_dir,
          warmstart=cmd_args.warmstart)
