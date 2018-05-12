from headers import *
import common
import utils

import sys, os, platform, pickle, json, argparse, time

import numpy as np
import random

from HRL.eval_motion import create_motion
from HRL.RNNController import RNNPlanner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def learn_controller(args):

    elap = time.time()

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    if args['object_target']:
        common.ensure_object_targets()

    set_seed(args['seed'])
    task = common.create_env(args['house'], task_name=args['task_name'], false_rate=args['false_rate'],
                             success_measure=args['success_measure'],
                             depth_input=args['depth_input'],
                             target_mask_input=args['target_mask_input'],
                             segment_input=args['segmentation_input'],
                             cacheAllTarget=True,
                             render_device=args['render_gpu'],
                             use_discrete_action=True,
                             include_object_target=args['object_target'],
                             include_outdoor_target=args['outdoor_target'],
                             discrete_angle=True)

    # create motion
    __controller_warmstart = args['warmstart']
    args['warmstart'] = args['motion_warmstart']
    motion = create_motion(args, task)
    args['warmstart'] = __controller_warmstart

    # logger
    logger = utils.MyLogger(args['save_dir'], True)

    logger.print("> Planner Units = {}".format(args['units']))
    logger.print("> Max Planner Steps = {}".format(args['max_planner_steps']))
    logger.print("> Max Exploration Steps = {}".format(args['max_exp_steps']))
    logger.print("> Reward = {} & {}".format(args['time_penalty'], args['success_reward']))

    # Planner Learning
    logger.print('Start RNN Planner Learning ...')

    planner = RNNPlanner(motion, args['units'], args['warmstart'])

    fixed_target = None
    if args['only_eval_room']:
        fixed_target = 'any-room'
    elif args['only_eval_object']:
        fixed_target = 'any-object'
    train_stats, eval_stats = \
        planner.learn(args['iters'], args['max_episode_len'],
                      target=fixed_target,
                      motion_steps=args['max_exp_steps'],
                      planner_steps=args['max_planner_steps'],
                      batch_size=args['batch_size'],
                      lrate=args['lrate'], grad_clip=args['grad_clip'],
                      weight_decay=args['weight_decay'], gamma=args['gamma'],
                      entropy_penalty=args['entropy_penalty'],
                      save_dir=args['save_dir'],
                      report_rate=5, eval_rate=20, save_rate=100,
                      logger=logger, seed=args['seed'])

    logger.print('######## Done ###########')
    filename = args['save_dir']
    if filename[-1] != '/': filename = filename + '/'
    filename = filename + 'train_stats.pkl'
    with open(filename, 'wb') as f:
        pickle.dump([train_stats, eval_stats], f)
    logger.print('  --> Training Stats Saved to <{}>!'.format(filename))
    return planner


def parse_args():
    parser = argparse.ArgumentParser("Learning Bayes Graph for 3D House Navigation")
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    parser.add_argument("--false-rate", type=float, default=0, help='The Rate of Impossible Targets')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--render-gpu", type=int, help="gpu id for rendering the environment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
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
    parser.add_argument("--success-measure", choices=['center', 'stay', 'see'], default='see',
                        help="criteria for a successful episode")
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also a target. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--no-outdoor-target", dest='outdoor_target', action='store_false',
                        help="when this flag is set, we will exclude <outdoor> target")
    parser.set_defaults(outdoor_target=True)
    parser.add_argument("--only-eval-room-target", dest='only_eval_room', action='store_true',
                        help="when this flag is set, only evaluate room targets. only effective when --include-object-target")
    parser.set_defaults(only_eval_room=False)
    parser.add_argument("--only-eval-object-target", dest='only_eval_object', action='store_true',
                        help="when this flag is set, only evaluate object targets. only effective when --include-object-target")
    parser.set_defaults(only_eval_object=False)
    # Core learning parameters
    parser.add_argument("--units", type=int, default=50, help="rnn units in the planner")
    parser.add_argument("--iters", type=int, default=10000, help="training iterations")
    parser.add_argument("--max-episode-len", type=int, default=300, help="maximum episode length")
    parser.add_argument("--max-exp-steps", type=int, default=30, help="maximum number of allowed exploration steps")
    parser.add_argument("--max-planner-steps", type=int, default=10, help="maximum number of allowed planner steps")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lrate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay in optimizer")
    parser.add_argument("--grad-clip", type=float, default=1, help="gradient clipping in optimizer")
    parser.add_argument("--entropy-penalty", type=float, default=0.1, help="entropy bonus")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--time-penalty", type=float, default=0.1, help="penalty reward reward per planner step")
    parser.add_argument("--success-reward", type=float, default=2, help="success reward")
    # Core policy parameters
    parser.add_argument("--motion", choices=['rnn', 'fake', 'random', 'mixture'], default="fake", help="type of the locomotion")
    parser.add_argument("--random-motion-skill", type=int, default=6, help="skill rate for random motion, only effective when --motion random")
    parser.add_argument("--mixture-motion-dict", type=str, help="dict for mixture-motion, only effective when --motion mixture")
    parser.add_argument("--motion-warmstart", type=str, help="file to load the policy parameters")
    parser.add_argument("--motion-warmstart-dict", type=str, dest='warmstart_dict',
                        help="arg dict the policy model, only effective when --motion rnn")
    parser.add_argument("--terminate-measure", choices=['mask', 'stay', 'see'], default='mask',
                        help="criteria for terminating a motion execution")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'],
                        help="[RNN-Only] RNN cell type")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_graph_/controller", help="directory in which graph parameters and logs will be stored")
    parser.add_argument("--warmstart", type=str, help="file to load a pre-trained graph")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is None) or (os.path.exists(args.warmstart)), 'Graph File Not Exists!'
    assert (args.motion_warmstart is None) or (os.path.exists(args.motion_warmstart)), 'Policy Model File Not Exists!'

    common.set_house_IDs(args.env_set)
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    if not os.path.exists(args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.motion not in ['fake', 'random']:
        assert args.motion_warmstart is not None

    if args.motion == 'mixture':
        assert args.mixture_motion_dict is not None

    if args.seed is None:
        args.seed = 0

    dict_args = args.__dict__

    # store training args
    filename = args.save_dir
    if filename[-1] != '/':
        filename = filename + '/'
    filename = filename + 'learning_args.json'
    with open(filename, 'w') as f:
        json.dump(dict_args, f)

    learn_controller(dict_args)
