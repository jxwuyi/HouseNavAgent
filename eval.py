from headers import *
import common
import utils

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def proc_info(info):
    return dict(pos=(info['pos'].x, info['pos'].y, info['pos'].z),
                yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])

def evaluate(house, seed = 0,
             iters = 1000, max_episode_len = 1000,
             hardness = None, success_measure = 'center', multi_target=False, fixed_target=None,
             algo='nop', model_name='cnn',
             model_file=None, log_dir='./log/eval',
             store_history=False, use_batch_norm=True,
             rnn_units=None, rnn_layers=None, rnn_cell=None,
             use_action_gating=False, use_residual_critic=False, use_target_gating=False,
             segmentation_input='none', depth_input=False, resolution='normal', history_len=4):

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    args = common.create_default_args(algo, model=model_name, use_batch_norm=use_batch_norm,
                                      replay_buffer_size=50,
                                      episode_len=max_episode_len,
                                      rnn_units=rnn_units, rnn_layers=rnn_layers, rnn_cell=rnn_cell,
                                      segmentation_input=segmentation_input,
                                      resolution_level=resolution,
                                      depth_input=depth_input,
                                      history_frame_len=history_len)
    args['action_gating'] = use_action_gating
    args['residual_critic'] = use_residual_critic
    args['multi_target'] = multi_target
    args['target_gating'] = use_target_gating

    if model_name == 'rnn':
        import zmq_train
        trainer = zmq_train.create_zmq_trainer(algo, model_name, args)
    else:
        trainer = common.create_trainer(algo, model_name, args)
    if model_file is not None:
        trainer.load(model_file)
    trainer.eval()  # evaluation mode

    if hardness is not None:
        print('>>>> Hardness = {}'.format(hardness))
    set_seed(seed)
    env = common.create_env(house, hardness=hardness, success_measure=success_measure,
                            depth_input=depth_input,
                            segment_input=args['segment_input'])

    if fixed_target is not None:
        env.reset_target(fixed_target)
    flag_random_reset_target = multi_target and (fixed_target is None)

    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    elap = time.time()
    t = 0
    for it in range(iters):
        cur_infos = []
        trainer.reset_agent()
        set_seed(seed + it + 1)  # reset seed
        obs = env.reset(reset_target=flag_random_reset_target)
        target_id = common.target_instruction_dict[env.get_current_target()]
        if multi_target and hasattr(trainer, 'set_target'):
            trainer.set_target(env.get_current_target())
        if store_history:
            cur_infos.append(proc_info(env.cam_info))
            #cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
        if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
        episode_success.append(0)
        episode_good.append(0)
        cur_stats = dict(best_dist=1e50,
                         success=0, good=0, reward=0, target=env.get_current_target(),
                         length=max_episode_len, images=None)
        if hasattr(env.world, "_id"):
            cur_stats['world_id'] = env.world._id
        episode_step = 0
        for _st in range(max_episode_len):
            # get action
            if trainer.is_rnn():
                idx = 0
                if multi_target:
                    action, _ = trainer.action(obs, return_numpy=True, target=[[target_id]])
                else:
                    action, _ = trainer.action(obs, return_numpy=True)
                action = int(action.squeeze())
            else:
                idx = trainer.process_observation(obs)
                action = trainer.action(True)  # use gumbel noise
            # environment step
            obs, rew, done, info = env.step(action)
            if store_history:
                cur_infos.append(proc_info(info))
                #cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
            if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
            cur_dist = env.cam_info['dist']
            if cur_dist == 0:
                cur_stats['good'] += 1
                episode_good[-1] = 1
            t += 1
            if cur_dist < cur_stats['best_dist']:
                cur_stats['best_dist'] = cur_dist
            episode_step += 1
            # collect experience
            trainer.process_experience(idx, action, rew, done, (_st + 1 >= max_episode_len), info)
            if done:
                episode_success[-1] = 1
                cur_stats['success'] = 1
                cur_stats['length'] = episode_step
                break
        if store_history:
            cur_stats['infos'] = cur_infos
        episode_stats.append(cur_stats)

        dur = time.time() - elap
        logger.print('Episode#%d, Elapsed = %.3f min' % (it+1, dur/60))
        if multi_target:
            logger.print('  ---> Target Room = {}'.format(cur_stats['target']))
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
    if multi_target:
        all_targets = list(set([s['target'] for s in episode_stats]))
        for tar in all_targets:
            n = sum([1.0 for s in episode_stats if s['target'] == tar])
            succ = [float(s['success'] > 0) for s in episode_stats if s['target'] == tar]
            good = [float(s['good'] > 0) for s in episode_stats if s['target'] == tar]
            length = [s['length'] for s in episode_stats if s['target'] == tar]
            good_len = np.mean([l for l,g in zip(length, good) if g > 0.5])
            succ_len = np.mean([l for l,s in zip(length, succ) if s > 0.5])
            logger.print(
                '>>>>> Multi-Target <%s>: Rate = %.3f (n=%d), Good = %.3f (AvgLen=%.3f), Succ = %.3f (AvgLen=%.3f)'
                % (tar, n/len(episode_stats), n, np.mean(good), good_len, np.mean(succ), succ_len))


    return episode_stats


def render_episode(env, images):
    for im in images:
        env.render(im)
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser("Evaluation for 3D House Navigation")
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--action-dim", type=int, help="degree of freedom of the agent movement, default=4, must be in range of [2,4]")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'], default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--history-frame-len", type=int, default=4,
                        help="length of the stacked frames, default=4")
    parser.add_argument("--success-measure", choices=['center', 'stay', 'see'], default='center',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--fixed-target", choices=common.all_target_instructions,
                        help="once set, all the episode will be fixed to a specific target.")
    # Core parameters
    parser.add_argument("--algo", choices=['ddpg','pg', 'rdpg', 'ddpg_joint', 'ddpg_alter', 'ddpg_eagle',
                                           'a2c', 'qac', 'dqn', 'nop', 'a3c'], default="ddpg", help="algorithm for training")
    parser.add_argument("--max-episode-len", type=int, default=2000, help="maximum episode length")
    parser.add_argument("--max-iters", type=int, default=1000, help="maximum number of eval episodes")
    parser.add_argument("--store-history", action='store_true', default=False, help="whether to store all the episode frames")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--use-action-gating", dest='action_gating', action='store_true',
                        help="whether to use action gating structure in the critic model")
    parser.set_defaults(action_gating=False)
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)
    parser.add_argument("--use-residual-critic", dest='residual_critic', action='store_true',
                        help="whether to use residual structure for feature extraction in the critic model (N.A. for joint-ac model) ")
    parser.set_defaults(residual_critic=False)
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'],
                        help="[RNN-Only] RNN cell type")
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    parser.add_argument("--warmstart", type=str, help="file to load the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is None) or (os.path.exists(args.warmstart)), 'Model File Not Exists!'

    common.set_house_IDs(args.env_set)
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    if not os.path.exists(args.log_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.log_dir))
        os.makedirs(args.log_dir)

    if args.action_dim is not None:
        common.action_shape = (args.action_dim, 2)
        print('degree of freedom of the action set to <{}>'.format(args.action_dim))

    if args.warmstart is None:
        model_name = 'random'
    elif args.algo in ['a2c', 'a3c']:
        model_name = 'rnn'
    else:
        model_name = 'cnn'
    episode_stats = \
        evaluate(args.house, args.seed or 0, args.max_iters, args.max_episode_len,
                 args.hardness, args.success_measure, args.multi_target, args.fixed_target,
                 args.algo, model_name, args.warmstart, args.log_dir,
                 args.store_history, args.use_batch_norm,
                 args.rnn_units, args.rnn_layers, args.rnn_cell,
                 args.action_gating, args.residual_critic, args.target_gating,
                 args.segmentation_input, args.depth_input, args.resolution, args.history_frame_len)

    if args.store_history:
        filename = args.log_dir
        if filename[-1] != '/':
            filename += '/'
        filename += args.algo+'_full_eval_history.pkl'
        print('Saving all stats to <{}> ...'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([episode_stats, args], f)
        print('  >> Done!')
