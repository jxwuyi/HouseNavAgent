from headers import *
import common
import utils

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def proc_info(info):
    return dict(pos=(info['pos'].x, info['pos'].y, info['pos'].z),
                yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])


def evaluate_aux_pred(house, seed = 0,iters = 1000, max_episode_len = 10,
                      algo='a3c', model_name='rnn', model_file=None, log_dir='./log/eval',
                      store_history=False, use_batch_norm=True,
                      rnn_units=None, rnn_layers=None, rnn_cell=None,
                      multi_target=True, use_target_gating=False,
                      segmentation_input='none', depth_input=False, resolution='normal'):

    # TODO: currently do not support this
    assert False, 'Aux Prediction Not Supported!'

    # Do not need to log detailed computation stats
    assert algo in ['a3c', 'nop']
    flag_run_random_policy = (algo == 'nop')
    common.debugger = utils.FakeLogger()
    args = common.create_default_args(algo, model=model_name, use_batch_norm=use_batch_norm,
                                      replay_buffer_size=50,
                                      episode_len=max_episode_len,
                                      rnn_units=rnn_units, rnn_layers=rnn_layers, rnn_cell=rnn_cell,
                                      segmentation_input=segmentation_input,
                                      resolution_level=resolution,
                                      depth_input=depth_input,
                                      history_frame_len=1)
    # TODO: add code for evaluation aux-task (concept learning)
    args['multi_target'] = multi_target
    args['target_gating'] = use_target_gating
    args['aux_task'] = True
    import zmq_train
    trainer = zmq_train.create_zmq_trainer(algo, model_name, args)
    if model_file is not None:
        trainer.load(model_file)
    trainer.eval()  # evaluation mode
    set_seed(seed)
    env = common.create_env(house, hardness=1e-8, success_measure='stay',
                            depth_input=depth_input,
                            segment_input=args['segment_input'],
                            genRoomTypeMap=True,
                            cacheAllTarget=True,
                            use_discrete_action=True)

    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating Auxiliary Task ...')
    logger.print('  --> Episode (Left) Turning Steps = {}'.format(max_episode_len))
    episode_err = []
    episode_succ = []
    episode_good = []
    episode_rews = []
    episode_stats = []
    elap = time.time()

    for it in range(iters):
        trainer.reset_agent()
        set_seed(seed + it + 1)  # reset seed
        obs = env.reset() if multi_target else env.reset(target=env.get_current_target())
        target_id = common.target_instruction_dict[env.get_current_target()]
        if multi_target and hasattr(trainer, 'set_target'):
            trainer.set_target(env.get_current_target())
        cur_infos = []
        if store_history:
            cur_infos.append(proc_info(env.info))
            # cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
        if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
        episode_succ.append(0)
        episode_err.append(0)
        episode_good.append(0)
        cur_rew = []
        cur_pred = []
        if flag_run_random_policy:
            predefined_aux_pred = common.all_aux_predictions[random.choice(common.all_target_instructions)]
        for _st in range(max_episode_len):
            # get action
            if flag_run_random_policy:
                aux_pred = predefined_aux_pred
            else:
                if multi_target:
                    _, _, aux_prob = trainer.action(obs, return_numpy=True, target=[[target_id]],
                                                    return_aux_pred=True, return_aux_logprob=False)
                else:
                    _, _, aux_prob = trainer.action(obs, return_numpy=True, return_aux_pred=True, return_aux_logprob=False)
                aux_prob = aux_prob.squeeze()  # [n_pred]
                aux_pred = int(np.argmax(aux_prob))  # greedy action, takes the output with the maximum confidence
            aux_rew = trainer.get_aux_task_reward(aux_pred, env.get_current_room_pred_mask())
            cur_rew.append(aux_rew)
            cur_pred.append(common.all_aux_prediction_list[aux_pred])
            if aux_rew < 0:
                episode_err[-1] += 1
            if aux_rew >= 0.9:  # currently a hack
                episode_succ[-1] += 1
            if aux_rew > 0:
                episode_good[-1] += 1
            action = 5  # Left Rotation
            # environment step
            obs, rew, done, info = env.step(action)
            if store_history:
                cur_infos.append(proc_info(info))
                cur_infos[-1]['aux_pred'] = cur_pred
                #cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
            if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
        if episode_err[-1] > 0:
            episode_succ[-1] = 0
        room_mask = env.get_current_room_pred_mask()
        cur_room_types = []
        for i in range(common.n_aux_predictions):
            if (room_mask & (1 << i)) > 0:
                cur_room_types.append(common.all_aux_prediction_list[i])

        cur_stats = dict(err=episode_err[-1], good=episode_good[-1], succ=episode_succ[-1], rew=cur_rew,
                         err_rate=episode_err[-1]/max_episode_len,
                         good_rate=episode_good[-1]/max_episode_len,
                         succ_rate=episode_succ[-1]/max_episode_len,
                         target=env.get_current_target(),
                         mask=room_mask,
                         room_types=cur_room_types,
                         length=max_episode_len)
        if store_history:
            cur_stats['infos'] = cur_infos
        episode_stats.append(cur_stats)

        dur = time.time() - elap
        logger.print('Episode#%d, Elapsed = %.3f min' % (it+1, dur/60))
        logger.print('  ---> Target Room = {}'.format(cur_stats['target']))
        logger.print('  ---> Aux Rew = {}'.format(cur_rew))
        if (episode_succ[-1] > 0) and (episode_err[-1] == 0):
            logger.print('  >>>> Success!')
        elif episode_err[-1] == 0:
            logger.print('  >>>> Good!')
        else:
            logger.print('  >>>> Failed!')
        logger.print("  ---> Indep. Prediction: Succ Rate = %.3f, Good Rate = %.3f, Err Rate = %.3f"
                     % (episode_succ[-1] * 100.0 / max_episode_len,
                        episode_good[-1] * 100.0 / max_episode_len,
                        episode_err[-1] * 100.0 / max_episode_len))
        logger.print("  > Accu. Succ = %.3f, Good = %.3f, Fail = %.3f"
                     % (float(np.mean([float(s == max_episode_len) for s in episode_succ])) * 100.0,
                        float(np.mean([float(e == 0) for e in episode_err])) * 100,
                        float(np.mean([float(e > 0) for e in episode_err])) * 100))
        logger.print("  > Accu. Rate: Succ Rate = %.3f, Good Rate = %.3f, Fail Rate = %.3f"
                     % (float(np.mean([s / max_episode_len for s in episode_succ])) * 100.0,
                        float(np.mean([g / max_episode_len for g in episode_good])) * 100,
                        float(np.mean([e / max_episode_len for e in episode_err])) * 100))
    return episode_stats


def evaluate(house, seed = 0, render_device=None,
             iters = 1000, max_episode_len = 1000,
             hardness = None, success_measure = 'center', multi_target=False, fixed_target=None,
             algo='nop', model_name='cnn',
             model_file=None, log_dir='./log/eval',
             store_history=False, use_batch_norm=True,
             rnn_units=None, rnn_layers=None, rnn_cell=None,
             use_action_gating=False, use_residual_critic=False, use_target_gating=False,
             segmentation_input='none', depth_input=False, resolution='normal', history_len=4,
             include_object_target=False,
             aux_task=False, no_skip_connect=False, feed_forward=False,
             greedy_execution=False, greedy_aux_pred=False):

    assert not aux_task, 'Do not support Aux-Task now!'

    elap = time.time()

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
    args['object_target'] = include_object_target
    args['target_gating'] = use_target_gating
    args['aux_task'] = aux_task
    args['no_skip_connect'] = no_skip_connect
    args['feed_forward'] = feed_forward
    if fixed_target is not None:
        assert fixed_target in common.n_target_instructions, 'invalid fixed target <{}>'.format(fixed_target)

    if model_name == 'rnn':
        import zmq_train
        trainer = zmq_train.create_zmq_trainer(algo, model_name, args)
    else:
        trainer = common.create_trainer(algo, model_name, args)
    if model_file is not None:
        trainer.load(model_file)
    trainer.eval()  # evaluation mode
    if greedy_execution and hasattr(trainer, 'set_greedy_execution'):
        trainer.set_greedy_execution()
    else:
        print('[Eval] WARNING!!! Greedy Policy Execution NOT Available!!!')
        greedy_execution = False
    if greedy_aux_pred and hasattr(trainer, 'set_greedy_aux_prediction'):
        trainer.set_greedy_aux_prediction()
    else:
        print('[Eval] WARNING!!! Greedy Execution of Auxiliary Task NOT Available!!!')
        greedy_aux_pred = False

    if aux_task: assert trainer.is_rnn()  # only rnn support aux_task

    if hardness is not None:
        print('>>>> Hardness = {}'.format(hardness))
    set_seed(seed)
    env = common.create_env(house, hardness=hardness, success_measure=success_measure,
                            depth_input=depth_input,
                            segment_input=args['segment_input'],
                            genRoomTypeMap=aux_task,
                            cacheAllTarget=multi_target,
                            render_device=render_device,
                            include_object_target=include_object_target)

    if fixed_target is not None:
        env.reset_target(fixed_target)
    flag_random_reset_target = multi_target and (fixed_target is None)

    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    t = 0
    for it in range(iters):
        cur_infos = []
        trainer.reset_agent()
        set_seed(seed + it + 1)  # reset seed
        obs = env.reset(target=fixed_target)
        #if multi_target and (fixed_target is not None) and (fixed_target != 'kitchen'):
        #    # TODO: Currently a hacky solution
        #    env.reset(target=fixed_target)
        #    if house < 0:  # multi-house env
        #        obs = env.reset(reset_target=False, keep_world=True)
        #    else:
        #        obs = env.reset(reset_target=False)
        #else:
        #    # TODO: Only support multi-target + fixed kitchen; or fixed-target (kitchen)
        #    obs = env.reset(reset_target=flag_random_reset_target)
        target_id = common.target_instruction_dict[env.get_current_target()]
        if multi_target and hasattr(trainer, 'set_target'):
            trainer.set_target(env.get_current_target())
        if store_history:
            cur_infos.append(proc_info(env.info))
            #cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
        if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
        episode_success.append(0)
        episode_good.append(0)
        cur_stats = dict(best_dist=1e50,
                         success=0, good=0, reward=0, target=env.get_current_target(),
                         length=max_episode_len, images=None)
        if aux_task:
            cur_stats['aux_pred_rew'] = 0
            cur_stats['aux_pred_err'] = 0
        if hasattr(env.house, "_id"):
            cur_stats['world_id'] = env.house._id
        episode_step = 0
        for _st in range(max_episode_len):
            # get action
            if trainer.is_rnn():
                idx = 0
                if multi_target:
                    if aux_task:
                        action, _, aux_pred = trainer.action(obs, return_numpy=True, target=[[target_id]], return_aux_pred=True)
                    else:
                        action, _ = trainer.action(obs, return_numpy=True, target=[[target_id]])
                else:
                    if aux_task:
                        action, _, aux_pred = trainer.action(obs, return_numpy=True, return_aux_pred=True)
                    else:
                        action, _ = trainer.action(obs, return_numpy=True)
                action = action.squeeze()
                if greedy_execution:
                    action = int(np.argmax(action))
                else:
                    action = int(action)
                if aux_task:
                    aux_pred = aux_pred.squeeze()
                    if greedy_aux_pred:
                        aux_pred = int(np.argmax(aux_pred))
                    else:
                        aux_pred = int(aux_pred)
                    aux_rew = trainer.get_aux_task_reward(aux_pred, env.get_current_room_pred_mask())
                    cur_stats['aux_pred_rew'] += aux_rew
                    if aux_rew < 0: cur_stats['aux_pred_err'] += 1
            else:
                idx = trainer.process_observation(obs)
                action = trainer.action(None if greedy_execution else 1.0)  # use gumbel noise
            # environment step
            obs, rew, done, info = env.step(action)
            if store_history:
                cur_infos.append(proc_info(info))
                #cur_images.append(env.render(renderMapLoc=env.cam_info['loc'], display=False))
            if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
            cur_dist = info['dist']
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
                if rew > 5:  # magic number:
                    episode_success[-1] = 1
                    cur_stats['success'] = 1
                cur_stats['length'] = episode_step
                if aux_task:
                    cur_stats['aux_pred_err'] /= episode_step
                    cur_stats['aux_pred_rew'] /= episode_step
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
        if aux_task:
            logger.print('    >>>>>> Aux-Task: Avg Rew = %.4f, Avg Err = %.4f' % (cur_stats['aux_pred_rew'], cur_stats['aux_pred_err']))

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
    if aux_task:
        logger.print(' -->>> Auxiliary-Task: Mean Episode Avg Rew = %.6f, Mean Episode Avg Err = %.6f'
                     % (np.mean([float(s['aux_pred_rew']) for s in episode_stats]),
                        np.mean([float(s['aux_pred_err']) for s in episode_stats])))

    return episode_stats


def render_episode(env, images):
    for im in images:
        env.show(im)
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser("Evaluation for 3D House Navigation")
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--render-gpu", type=int, help="gpu id for rendering the environment")
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
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also a target. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--fixed-target", choices=common.ALLOWED_TARGET_ROOM_TYPES + common.ALLOWED_OBJECT_TARGET_TYPES,
                        help="once set, all the episode will be fixed to a specific target.")
    parser.add_argument("--greedy-execution", dest='greedy_execution', action='store_true',
                        help="When --greedy-execution, we directly take the action with the maximum probability instead of sampling. For DDPG, we turn off the gumbel-noise. For NOP, we will use discrete actions.")
    parser.set_defaults(greedy_execution=False)
    parser.add_argument("--greedy-aux-prediction", dest='greedy_aux_pred', action='store_true',
                        help="[A3C-Aux-Task-Only] When --greedy-execution, we directly take the auxiliary prediction with the maximum probability instead of sampling")
    parser.set_defaults(greedy_aux_pred=False)
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
    # Auxiliary Task Options
    parser.add_argument("--auxiliary-task", dest='aux_task', action='store_true',
                        help="Whether to perform auxiliary task of predicting room types")
    parser.set_defaults(aux_task=False)
    # Ablation Test Options
    parser.add_argument("--no-skip-connect", dest='no_skip_connect', action='store_true',
                        help="[A3C-LSTM Only] no skip connect. only takes the output of rnn to compute action")
    parser.set_defaults(no_skip_connect=False)
    parser.add_argument("--feed-forward-a3c", dest='feed_forward', action='store_true',
                        help="[A3C-LSTM Only] skip rnn completely. essentially cnn-a3c")
    parser.set_defaults(feed_forward=False)
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    parser.add_argument("--warmstart", type=str, help="file to load the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is None) or (os.path.exists(args.warmstart)), 'Model File Not Exists!'

    if args.aux_task:
        assert args.algo == 'a3c', 'Auxiliary Task is only supprted for <--algo a3c>'

    common.set_house_IDs(args.env_set, ensure_kitchen=(not args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    if args.object_target:
        common.ensure_object_targets()

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

    if args.hardness <= 1e-6:
        assert args.aux_task, 'When Hardness == 0, option --auxiliary-task must be set!'
        episode_stats = evaluate_aux_pred(args.house, args.seed or 0, args.max_iters, args.max_episode_len,
                                          args.algo, model_name, args.warmstart, args.log_dir, args.store_history,
                                          args.use_batch_norm,
                                          args.rnn_units, args.rnn_layers, args.rnn_cell,
                                          args.multi_target, args.target_gating,
                                          args.segmentation_input, args.depth_input, args.resolution)
    else:
        episode_stats = \
            evaluate(args.house, args.seed or 0, args.render_gpu, args.max_iters, args.max_episode_len,
                     args.hardness, args.success_measure, args.multi_target, args.fixed_target,
                     args.algo, model_name, args.warmstart, args.log_dir,
                     args.store_history, args.use_batch_norm,
                     args.rnn_units, args.rnn_layers, args.rnn_cell,
                     args.action_gating, args.residual_critic, args.target_gating,
                     args.segmentation_input, args.depth_input, args.resolution, args.history_frame_len,
                     include_object_target=args.object_target,
                     aux_task=args.aux_task, no_skip_connect=args.no_skip_connect, feed_forward=args.feed_forward,
                     greedy_execution=(args.greedy_execution and (args.algo == 'a3c')),
                     greedy_aux_pred=(args.greedy_aux_pred and (args.algo == 'a3c') and args.aux_task))

    if args.store_history:
        filename = args.log_dir
        if filename[-1] != '/':
            filename += '/'
        filename += args.algo+'_full_eval_history.pkl'
        print('Saving all stats to <{}> ...'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([episode_stats, args], f)
        print('  >> Done!')
