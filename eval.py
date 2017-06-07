from headers import *
import common
import utils

import os, sys, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(iters = 1000, max_episode_len = 2000, algo='nop',
            model_name='random', model_file=None, log_dir='./temp/eval',
            store_images=False):
    args = common.create_default_args(algo)
    trainer = common.create_trainer(algo,model_name,args)
    if model_file is not None:
        trainer.load(model_file)
    env = common.create_env()
    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_best_dist = []
    episode_success = []
    episode_len = []
    all_images = []
    elap = time.time()
    t = 0
    for it in range(iters):
        cur_images = []
        obs = env.reset()
        if store_images:
            cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
        obs = obs.transpose([1, 0, 2])
        episode_best_dist.append(env.cam_info['dist'])
        episode_success.append(0)
        episode_len.append(max_episode_len)
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
            t += 1
            if cur_dist < episode_best_dist[-1]:
                episode_best_dist[-1] = cur_dist
            episode_step += 1
            # collect experience
            trainer.process_experience(idx, action, rew, done, False)
            if done:
                obs = env.reset()
                obs = obs.transpose([1, 0, 2])
                episode_success[-1] = 1
                episode_len[-1] = episode_step
                break
        if store_images:
            all_images.append(cur_images)

        logger.print('Episode#%d, Elapsed = %.3f min' % (it+1, (time.time()-elap)/ 60))
        logger.print('  ---> Total Samples = {}'.format(t))
        logger.print('  ---> Success = %d  (rate = %.3f), Best Dist = %.2f'
            % (episode_success[-1], np.mean(episode_success), episode_best_dist[-1]))

    return episode_success, episode_len, episode_best_dist, all_images

def render_episode(env, images):
    for im in images:
        env.render(im)
        time.sleep(0.5)

if __name__ == '__main__':
    ep_succ, ep_len, ep_best_dist, all_images = evaluate()

    print('Total Success Rate = {}'.format(np.mean(ep_succ)))
    print('Avg Success Steps = {}'.format(np.mean(ep_len)))

    if len(all_images) > 0:
        env = common.create_env()
        for i, images in enumerate(all_images):
            input('press any key to start rendering episode #{}'.format(i))
            env.reset_render()
            render_episode(images)
