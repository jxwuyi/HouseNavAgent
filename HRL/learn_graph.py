from headers import *
import common
import utils

import sys, os, platform, pickle, json, argparse, time

import numpy as np
import random

from HRL.eval_motion import create_motion
from HRL.BayesGraph import GraphPlanner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def learn_graph(args):

    elap = time.time()

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    common.ensure_object_targets(True)

    set_seed(args['seed'])
    task = common.create_env(args['house'], task_name=args['task_name'], false_rate=args['false_rate'],
                             success_measure=args['success_measure'],
                             depth_input=args['depth_input'],
                             target_mask_input=args['target_mask_input'],
                             segment_input=args['segmentation_input'],
                             cacheAllTarget=True,
                             render_device=args['render_gpu'],
                             use_discrete_action=True,
                             include_object_target=True,
                             include_outdoor_target=True,
                             discrete_angle=True)

    # create motion
    __graph_warmstart = args['warmstart']
    args['warmstart'] = args['motion_warmstart']
    motion = create_motion(args, task)

    # create graph
    args['warmstart'] = __graph_warmstart
    graph = GraphPlanner(motion)

    # logger
    logger = utils.MyLogger(args['save_dir'], True)

    logger.print("> Training Mode = {}".format(args['training_mode']))
    logger.print("> Graph Eps = {}".format(args['graph_eps']))
    logger.print("> N_Trials = {}".format(args['n_trials']))
    logger.print("> Max Exploration Steps = {}".format(args['max_exp_steps']))

    # Graph Building
    logger.print('Start Graph Building ...')

    if args['warmstart'] is not None:
        filename = args['warmstart']
        logger.print(' >>> Loading Pre-Trained Graph from {}'.format(filename))
        with open(filename, 'rb') as file:
            g_params = pickle.load(file)
        graph.set_parameters(g_params)

    train_mode = args['training_mode']
    if train_mode in ['mle', 'joint']:
        graph.learn(n_trial=args['n_trials'], max_allowed_steps=args['max_exp_steps'], eps=args['graph_eps'], logger=logger)

    if train_mode in ['evolution', 'joint']:
        graph.evolve()   # TODO: not implemented yet

    logger.print('######## Final Stats ###########')
    graph._show_prior_room(logger=logger)
    graph._show_prior_object(logger=logger)
    return graph


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
    parser.add_argument("--success-measure", choices=['center', 'stay', 'see'], default='center',
                        help="criteria for a successful episode")
    # Core learning parameters
    parser.add_argument("--training-mode", choices=['mle', 'evolution', 'joint'], default='mle',
                        help="training mode: Maximum-Likelihood-Estimate; Evolutional Methods (assume warmstart); MLE+Evolutional")
    parser.add_argument("--graph-eps", type=float, default=0.00001, help="eps of graph connectivity")
    parser.add_argument("--n-trials", type=int, default=25, help="number of trials for evaluating connectivity")
    parser.add_argument("--max-exp-steps", type=int, default=30, help="maximum number of allowed exploration steps")
    # Core policy parameters
    parser.add_argument("--motion", choices=['rnn', 'fake', 'random', 'mixture'], default="fake", help="type of the locomotion")
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
    parser.add_argument("--save-dir", type=str, default="./_graph_", help="directory in which graph parameters and logs will be stored")
    parser.add_argument("--warmstart", type=str, help="file to load a pre-trained graph")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is None) or (os.path.exists(args.warmstart)), 'Graph File Not Exists!'
    assert (args.motion_warmstart is None) or (os.path.exists(args.motion_warmstart)), 'Policy Model File Not Exists!'

    common.set_house_IDs(args.env_set)
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    common.ensure_object_targets()

    if not os.path.exists(args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.motion not in ['fake', 'random']:
        assert args.motion_warmstart is not None

    if args.seed is None:
        args.seed = 0

    dict_args = args.__dict__

    graph = learn_graph(dict_args)

    filename = args.save_dir
    if filename[-1] != '/':
        filename += '/'
    filename += args.training_mode + '_' + args.motion + '_graph_params.pkl'
    print('Saving Graph Parameters to <{}> ...'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(graph.parameters(), f)
    print('>>>> Done!')
