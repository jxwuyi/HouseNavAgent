from headers import *
import common
import utils

import os, sys, time, pickle, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(iters = 1000, max_episode_len = 1000, algo='nop',
             model_name='cnn', model_file=None, log_dir='./log/eval',
             store_images=False):
    args = common.create_default_args(algo)
    trainer = common.create_trainer(algo, model_name, args)
    if model_file is not None:
        trainer.load(model_file)
    env = common.create_env()
    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    elap = time.time()
    t = 0
    for it in range(iters):
        cur_images = []
        obs = env.reset()
        if store_images:
            cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
        obs = obs.transpose([1, 0, 2])
        episode_success.append(0)
        episode_good.append(0)
        cur_stats = dict(best_dist=1e50,
                         success=0, good=0, reward=0,
                         length=max_episode_len, images=None)
        episode_step = 0
        for _ in range(max_episode_len):
            idx = trainer.process_observation(obs)
            # get action
            action = trainer.action()
            # environment step
            obs, rew, done, info = env.step(action)
            if store_images:
                cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
            obs = obs.transpose([1, 0, 2])
            cur_dist = env.cam_info['dist']
            if cur_dist == 0:
                cur_stats['good'] += 1
                episode_good[-1] = 1
            t += 1
            if cur_dist < cur_stats['best_dist']:
                cur_stats['best_dist'] = cur_dist
            episode_step += 1
            # collect experience
            trainer.process_experience(idx, action, rew, done, False)
            if done:
                obs = env.reset()
                obs = obs.transpose([1, 0, 2])
                episode_success[-1] = 1
                cur_stats['success'] = 1
                cur_stats['length'] = episode_step
                break
        if store_images:
            cur_stats['images'] = cur_images
        episode_stats.append(cur_stats)

        dur = time.time() - elap
        logger.print('Episode#%d, Elapsed = %.3f min' % (it+1, dur/60))
        logger.print('  ---> Total Samples = {}'.format(t))
        logger.print('  ---> Success = %d  (rate = %.3f)'
                     % (cur_stats['success'], np.mean(episode_success)))
        logger.print('  ---> Times of Reaching Target Room = %d  (rate = %.3f)'
                     % (cur_stats['good'], np.mean(episode_good)))
        logger.print('  ---> Best Distance = %d' % cur_stats['best_dist'])

    logger.print('######## Final Stats ###########')
    logger.print('Success Rate = %.3f' % np.mean(episode_success))
    logger.print('Avg Length per Success = %.3f' % np.mean([s['length'] for s in episode_stats if s['success'] > 0]))
    logger.print('Reaching Target Rate = %.3f' % np.mean(episode_good))
    logger.print('Avg Length per Target Reach = %.3f' % np.mean([s['length'] for s in episode_stats if s['good'] > 0]))

    return episode_stats


def render_episode(env, images):
    for im in images:
        env.render(im)
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser("Evaluation for 3D House Navigation")
    # Environment
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # Core parameters
    parser.add_argument("--algo", type=str, default="ddpg", help="algorithm for training")
    parser.add_argument("--max-episode-len", type=int, help="maximum episode length")
    parser.add_argument("--max-iters", type=int, default=1000, help="maximum number of eval episodes")
    parser.add_argument("--store-history", action='store_true', default=False, help="whether to store all the episode frames")
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    parser.add_argument("--warmstart", type=str, help="file to load the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is not None) and (os.path.exists(args.warmstart)), 'Model File Not Exists!'

    if args.seed is not None:
        np.random.seed(args.seed)

    episode_stats = \
        evaluate(args.max_iters, args.max_episode_len,
                 args.algo, args.warmstart, args.log_dir,
                 args.store_history)

    if args.store_history:
        filename = args.log_dir
        if filename[-1] != '/':
            filename += '/'
        filename += args.algo+'_full_eval_history.pkl'
        print('Saving all stats to <{}> ...'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([episode_stats, args], f)
        print('  >> Done!')
