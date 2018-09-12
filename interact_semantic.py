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

from trainer.semantic import SemanticTrainer
from policy.cnn_classifier import CNNClassifier
from HRL.semantic_oracle import SemanticOracle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def proc_info(info):
    return dict(pos=(info['pos'].x, info['pos'].y, info['pos'].z),
                yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])



def evaluate(house, seed = 0, render_device=None, model_device=None,
             iters = 1000, max_episode_len = 1000,
             hardness = None, max_birthplace_steps=None, min_birthplace_grids=None,
             success_measure = 'see-stop', multi_target=False, fixed_target=None,
             model_dir=None, log_dir='./log/eval',
             store_history=False,
             segmentation_input='none', depth_input=False, 
             show_mask_feature=False,
             resolution='normal', 
             include_object_target=False,
             include_outdoor_target=True,
             cache_supervision=True,
             threshold=None):

    elap = time.time()
    
    # Process Observation Shape
    common.process_observation_shape(model='cnn',
                                     resolution_level=resolution,
                                     segmentation_input=segmentation_input,
                                     depth_input=depth_input,
                                     history_frame_len=1)

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()
    
    set_seed(seed)

    # load semantic classifiers
    print('Loading Semantic Oracle ...')
    oracle = SemanticOracle(model_dir=model_dir, model_device=model_device, include_object=False)

    # create env
    env = common.create_env(house,
                            hardness=hardness,
                            max_birthplace_steps=max_birthplace_steps,
                            min_birthplace_grids=min_birthplace_grids,
                            success_measure=success_measure,
                            reward_type='new',
                            depth_input=depth_input,
                            segment_input=segmentation_input,
                            cacheAllTarget=True,
                            render_device=render_device,
                            use_discrete_action=True,
                            include_object_target=include_object_target,
                            include_outdoor_target=include_outdoor_target,
                            target_mask_input=True,
                            discrete_angle=True,
                            cache_supervision=False)

    def display(obs):
        mask = oracle.get_mask_feature(obs, threshold=threshold)
        if threshold is not None:
            ret = [oracle.targets[i] for i in range(oracle.n_target) if mask[i] > 0]
        else:
            ret = [oracle.targets[i] + ": %.3f, " % mask[i] for i in range(oracle.n_target)]
        print('<Semantic>: {}'.format(ret))
        if show_mask_feature:
            mask_feat = env.get_feature_mask()
            env_ret = [oracle.targets[i] for i in range(oracle.n_target) if mask_feat[i] > 0]
            print('<<<Truth>>>: {}'.format(env_ret))


    if (fixed_target is not None) and ('any' not in fixed_target):
        env.reset_target(fixed_target)

    flag_random_reset_target = (fixed_target is None) or ('any' in fixed_target)

    logger = utils.MyLogger(log_dir, True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    t = 0

    # set actions
    action_dict = dict(i=0, u=1, o=2, a=3, f=4, k=5, s=6, d=7, p=8,r=-1,h=-2,q=-3)
    action_names = ['Forward', 'Left-Fwd', 'Right-Fwd', 'Left-Rotate', 'Right-Rotate',
                    'Small-Forward', 'Small-Left-Rot', 'Small-Right-Rot', 'Stay']

    def print_help():
        print('Usage: ')
        print('> Actions: Total 9 Actions')
        print('  --> i, k, u, o, p: forward, small-fwd, left-fwd, right-fwd, stay')
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
        target = env.info['target_room']

        def get_supervision_name(act):
            if act < 0: return 'N/A'
            discrete_action_names = ['Forward', 'Left-Fwd', 'Right-Fwd', 'Left-Rotate', 'Right-Rotate',
                                     'Small-Forward', 'Small-Left-Rot', 'Small-Right-Rot', 'Stay', 'Left', 'Right', 'Backward']
            return discrete_action_names[act]

        while True:
            print('Step#%d, Instruction = <go to %s>' % (step, target))
            if cache_supervision:
                print('  ->>>> supervision = {}'.format(get_supervision_name(env.info['supervision'])))
            mat = env.debug_show()
            mat = cv2.resize(mat, (800, 600), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("aaa", mat)
            #############
            # Plan Info
            #############
            display(obs)
            while True:
                key = cv2.waitKey(0)
                key = chr(key)
                if key in action_dict:
                    if key == 'h':
                        print_help()
                    else:
                        break
                else:
                    print('>> invalid key <{}>! press q to quit; r to reset; h to get helper'.format(key))
            if (key in action_dict) and (action_dict[key] < 0):
                break
            step += 1

            act = action_dict[key]

            obs, reward, done, info = env.step(act)
            
            rew += reward
            print('>> r = %.2f, done = %f, accu_rew = %.2f, step = %d' % (reward, done, rew, step))
            print('   info: collision = %d, raw_dist = %d, scaled_dist = %.3f, opt_steps = %d'
                    % (info['collision'], info['dist'], info['scaled_dist'], info['optsteps']))

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
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--render-gpu", type=int, help="gpu id for rendering the environment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
    parser.add_argument("--min-birthplace-grids", type=int, default=0,
                        help="int, the minimum grid distance of the birthplace towards target. Default 0, namely possible to born with gird_dist=0.")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'], default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--target-mask-input", dest='target_mask_input', action='store_true',
                        help="whether to include target mask 0/1 signal as part of the input signal")
    parser.set_defaults(target_mask_input=False)
    parser.add_argument("--success-measure", choices=['stay', 'see', 'see-stop'], default='see-stop',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also a target. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--no-outdoor-target", dest='outdoor_target', action='store_false',
                        help="when this flag is set, we will exclude <outdoor> target")
    parser.set_defaults(outdoor_target=True)
    parser.add_argument("--fixed-target", type=str,
                        help="once set, all the episode will be fixed to a specific target.")
    # Core parameters
    parser.add_argument("--max-episode-len", type=int, default=2000, help="maximum episode length")
    parser.add_argument("--max-iters", type=int, default=1000, help="maximum number of eval episodes")
    parser.add_argument("--store-history", action='store_true', default=False, help="whether to store all the episode frames")

    # Semantic Classifiers
    parser.add_argument('--semantic-dir', type=str, help='[SEMANTIC] root folder containing all semantic classifiers')
    parser.add_argument('--semantic-threshold', type=float, help='[SEMANTIC] threshold for semantic labels. None: probability')
    parser.add_argument("--semantic-gpu", type=int, help="[SEMANTIC] gpu id for running semantic classifier")

    # Other Options
    parser.add_argument("--no-cache-supervision", dest='cache_supervision', action='store_false',
                        help="When set, will not show supervision signal at each timestep (for saving caching time)")
    parser.set_defaults(cache_supervision=True)
    
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    common.set_house_IDs(args.env_set, ensure_kitchen=(not args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    common.ensure_object_targets(args.object_target)

    if args.fixed_target is not None:
        allowed_targets = list(common.target_instruction_dict.keys()) + ['any-room']
        if args.object_target:
            allowed_targets.append('any-object')
        assert args.fixed_target in allowed_targets, '--fixed-target specified an invalid target <{}>!'.format(cmd_args.fixed_target)
        if not ('any' in args.fixed_target):
            common.filter_house_IDs_by_target(args.fixed_target)
            print('[interact_semantic.py] Filter Houses By Fixed-Target <{}> to N=<{}> Houses...'.format(args.fixed_target, len(common.all_houseIDs)))

    if not os.path.exists(args.log_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.log_dir))
        os.makedirs(args.log_dir)

    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU.'
    if args.render_gpu is None:
        args.render_gpu = all_gpus[0]
    else:
        args.render_gpu = all_gpus[args.render_gpu]
    if args.semantic_gpu is None:
        args.semantic_gpu = all_gpus[0]
    else:
        args.semantic_gpu = all_gpus[args.semantic_gpu]

    evaluate(args.house, args.seed or 0, 
             render_device=args.render_gpu, model_device=args.semantic_gpu,
             iters=args.max_iters, max_episode_len=args.max_episode_len,
             hardness=args.hardness, max_birthplace_steps=args.max_birthplace_steps, min_birthplace_grids=args.min_birthplace_grids,
             success_measure=args.success_measure, multi_target=args.multi_target, fixed_target=args.fixed_target,
             model_dir=args.semantic_dir, log_dir=args.log_dir,
             store_history=args.store_history,
             segmentation_input=args.segmentation_input, depth_input=args.depth_input, 
             show_mask_feature=args.target_mask_input,
             resolution=args.resolution,
             include_object_target=args.object_target,
             include_outdoor_target=args.outdoor_target,
             cache_supervision=args.cache_supervision,
             threshold=args.semantic_threshold)
