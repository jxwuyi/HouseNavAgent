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



def evaluate(house, seed = 0, render_device=None,
             iters = 1000, max_episode_len = 1000,
             hardness = None, max_birthplace_steps=None,
             success_measure = 'center', multi_target=False, fixed_target=None,
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
    if (fixed_target is not None) and (fixed_target != 'any-room') and (fixed_target != 'any-object'):
        assert fixed_target in common.n_target_instructions, 'invalid fixed target <{}>'.format(fixed_target)

    if 'any' in fixed_target:
        common.ensure_object_targets(True)

    if hardness is not None:
        print('>>>> Hardness = {}'.format(hardness))
    if max_birthplace_steps is not None:
        print('>>>> Max_BirthPlace_Steps = {}'.format(max_birthplace_steps))
    set_seed(seed)
    env = common.create_env(house,
                            hardness=hardness,
                            max_birthplace_steps=max_birthplace_steps,
                            success_measure=success_measure,
                            reward_type='delta',
                            depth_input=depth_input,
                            segment_input=args['segment_input'],
                            genRoomTypeMap=aux_task,
                            cacheAllTarget=multi_target,
                            render_device=render_device,
                            use_discrete_action=('dpg' not in algo),
                            include_object_target=include_object_target)

    if (fixed_target is not None) and ('any' not in fixed_target):
        env.reset_target(fixed_target)

    # create model
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

    flag_random_reset_target = multi_target and (fixed_target is None)

    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    t = 0


    action_dict = dict(k=7,l=9,j=10,o=3,u=4,f=5,a=6,i=8,d=11,s=12,r=-1,h=-2,q=-3)
    policy_dict = dict(e=None, w=None,
                       z='sofa',x='chair',c='bed',v='toilet',b='table',n='dresser',m='vehicle')
    policy_dict['1']='kitchen'
    policy_dict['2']='living_room'
    policy_dict['3']='dining_room'
    policy_dict['4']='bedroom'
    policy_dict['5']='bathroom'
    policy_dict['6']='office'
    policy_dict['7']='garage'
    policy_dict['8']='outdoor'
    def print_help():
        print('Usage: ')
        print('> Subpolicies:')
        print('  --> e: continue previous sub-policy')
        print('  --> w: switch to original sub-policy')
        print('  --> 1 (kitchen), 2 (living room), 3 (dining room), 4 (bedroom), 5 (bathroom), 6 (office), 7 (garage), 8 (outdoor)')
        print('  --> z (sofa), x (chair), c (bed), v (toilet), b (table), n (dresser), m (vehicle)')
        print('> Actions: (Simplified Version) Total 10 Actions')
        print('  --> j, k, l, i, u, o: left, back, right, forward, left-forward, right-forward')
        print('  --> a, s, d, f: left-rotate, small left-rot, small right-rot, right-rotate')
        print('> press r: reset')
        print('> press h: show helper again')
        print('> press q: exit')

    import cv2
    print_help()
    eval = []
    while True:
        step = 0
        rew = 0
        good = 0
        obs = env.reset(target=fixed_target)
        if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
        target = env.info['target_room']
        cur_policy = target

        trainer.reset_agent()

        while True:
            print('Step#%d, Instruction = <go to %s>' % (step, target))
            mat = env.debug_show()
            mat = cv2.resize(mat, (800, 600), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("aaa", mat)
            while True:
                key = cv2.waitKey(0)
                key = chr(key)
                if key in action_dict:
                    if key == 'h':
                        print_help()
                    else:
                        break
                elif key in policy_dict:
                    if key == 'e':
                        print('Executing Sub-Policy <{}>'.format(cur_policy))
                        break
                    elif key == 'w':
                        cur_policy = target
                        print('Switch to Original Sub-Policy <{}>'.format(cur_policy))
                    else:
                        cur_policy = policy_dict[key]
                        print('Switch to Sub-Policy <{}>'.format(cur_policy))
                else:
                    print('>> invalid key <{}>! press q to quit; r to reset; h to get helper'.format(key))
            if (key in action_dict) and (action_dict[key] < 0):
                break
            step += 1

            if key in action_dict:
                action = action_dict[key]
            else:
                target_id = common.target_instruction_dict[cur_policy]
                if multi_target and hasattr(trainer, 'set_target'):
                    trainer.set_target(cur_policy)

                if trainer.is_rnn():
                    idx = 0
                    if multi_target:
                        action, _ = trainer.action(obs, return_numpy=True, target=[[target_id]])
                    else:
                        action, _ = trainer.action(obs, return_numpy=True)
                    action = action.squeeze()
                    if greedy_execution:
                        action = int(np.argmax(action))
                    else:
                        action = int(action)
                else:
                    idx = trainer.process_observation(obs)
                    action = trainer.action(None if greedy_execution else 1.0)  # use gumbel noise

            obs, reward, done, info = env.step(action)

            if model_name != 'rnn': obs = obs.transpose([1, 0, 2])
            rew += reward
            print('>> r = %.2f, done = %f, accu_rew = %.2f, step = %d' % (reward, done, rew, step))
            print('   info: collision = %d, raw_dist = %d, scaled_dist = %.3f, opt_steps = %d'
                    % (info['collision'], info['dist'], info['scaled_dist'], info['optsteps']))

            #############
            # Plan Info
            #############
            print('   S_aux = {}'.format(env.get_aux_tags()))
            print('   plan info: {}'.format(env.get_optimal_plan()))

            if done:
                good = 1
                print('Congratulations! You reach the Target!')
                print('>> Press any key to restart!')
                key = cv2.waitKey(0)
                break
        eval.append((step, rew, good))
        if key == 'q':
            break
    if len(eval) > 0:
        print('++++++++++ Task Stats +++++++++++')
        print("Episode Played: %d" % len(eval))
        succ = [e for e in eval if e[2] > 0]
        print("Success = %d, Rate = %.3f" % (len(succ), len(succ) / len(eval)))
        print("Avg Reward = %.3f" % (sum([e[1] for e in eval])/len(eval)))
        if len(succ) > 0:
            print("Avg Success Reward = %.3f" % (sum([e[1] for e in succ]) / len(succ)))
        print("Avg Step = %.3f" % (sum([e[0] for e in eval]) / len(eval)))
        if len(succ) > 0:
            print("Avg Success Step = %.3f" % (sum([e[0] for e in succ]) / len(succ)))



def render_episode(env, images):
    for im in images:
        env.show(im)
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser("Evaluation for 3D House Navigation")
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--render-gpu", type=int, help="gpu id for rendering the environment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
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
    parser.add_argument("--eval-target-type", choices=['all', 'only-room', 'only-object'], default='only-object',
                        help="the type of targets to evaluate on")
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

    assert args.object_target

    fixed_target = None
    if args.eval_target_type == 'only-room':
        fixed_target = 'any-room'
    elif args.eval_target_type == 'only-object':
        fixed_target = 'any-object'

    evaluate(args.house, args.seed or 0, 0, 10000, 10000,
             hardness=args.hardness, max_birthplace_steps=args.max_birthplace_steps,
             success_measure='see', multi_target=True,
             fixed_target=fixed_target,
             algo='a3c', model_name='rnn',
             model_file=args.warmstart, log_dir=args.log_dir,
             store_history=False, use_batch_norm=args.use_batch_norm,
             rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, rnn_cell=args.rnn_cell,
             use_action_gating=args.action_gating, use_residual_critic=args.residual_critic, use_target_gating=args.target_gating,
             segmentation_input=args.segmentation_input, depth_input=args.depth_input,
             resolution=args.resolution, history_len=args.history_frame_len,
             include_object_target=args.object_target,
             aux_task=False, no_skip_connect=args.no_skip_connect, feed_forward=args.feed_forward,
             greedy_execution=False)
