from headers import *
import common
import utils

import environment
import threading

from zmq_trainer.zmq_actor_critic import ZMQA3CTrainer
from zmq_trainer.zmq_util import ZMQSimulator, ZMQMaster
from zmq_trainer.zmqsimulator import SimulatorProcess, SimulatorMaster, ensure_proc_terminate

from policy.rnn_discrete_actor_critic import DiscreteRNNPolicy

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_scheduler(type='medium'):
    if type == 'none':
        return None
    if type == 'linear':
        return utils.LinearSchedule(200000, 1.0, 0.0)
    if type == 'medium':
        endpoints = [(0, 0), (2000, 0.1), (7000, 0.25), (40000, 0.5), (200000, 1.0)]
    elif type == 'high':
        endpoints = [(0, 0), (3000, 0.1), (15000, 0.25), (80000, 0.5), (500000, 1.0)]
    elif type == 'low': # low
        endpoints = [(0, 0), (1000, 0.1), (3000, 0.25), (20000, 0.5), (100000, 1.0)]
    elif type == 'tiny':  # low
        endpoints = [(0, 0), (1000, 0.1), (2000, 0.25), (5000, 0.5), (20000, 1.0)]
    elif type == 'exp':
        endpoints = [(0, 0), (1000, 0.01), (5000, 0.1), (10000, 0.5), (20000, 0.75), (50000, 0.9), (100000, 0.95), (200000, 1.0)]
    print('Building PiecewiseScheduler with <endpoints> = {}'.format(endpoints))
    scheduler = utils.PiecewiseSchedule(endpoints, outside_value=1.0)
    return scheduler


def create_policy(model_name, args, observation_shape, n_action):
    assert model_name == 'rnn', 'currently only support rnn policy!'
    model = DiscreteRNNPolicy(observation_shape, n_action,
                              conv_hiddens=[64, 64, 128, 128],
                              kernel_sizes=5, strides=2,
                              linear_hiddens=[512],
                              policy_hiddens=[128],
                              critic_hiddens=[32],
                              rnn_cell=args['rnn_cell'],
                              rnn_layers=args['rnn_layers'],
                              rnn_units=args['rnn_units'])
    if common.use_cuda:
        if 'train_gpu' in args:
            model.cuda(device_id=args['train_gpu'])
        else:
            model.cuda()
    return model


def create_zmq_trainer(algo, model, args):
    assert model == 'rnn', 'currently only support rnn policy!'
    observation_shape = common.observation_shape
    n_action = common.n_discrete_actions
    if algo == 'a3c':
        model_gen = lambda: create_policy(model, args, observation_shape, n_action)
        trainer = ZMQA3CTrainer('ZMQA3CTrainer', model_gen, observation_shape, [n_action], args)
    else:
        assert False, '[ZMQ Trainer] Trainer <{}> is not defined!'.format(algo)
    return trainer


def create_zmq_config(args):
    config = dict()

    # env param
    config['n_house'] = args['n_house']
    config['hardness'] = args['hardness']
    all_gpus = common.get_gpus_for_rendering()
    if 'render_gpu' in args:
        gpu_ids = args['render_gpu'].split(',')
        render_gpus = [all_gpus[int(k)] for k in gpu_ids]
    elif 'train_gpu' in args:
        k = args['train_gpu']
        render_gpus = all_gpus[:k] + all_gpus[k+1:]
    else:
        render_gpus = all_gpus[1:]
    config['render_devices'] = tuple(render_gpus)
    config['segment_input'] = args['segment_input']
    config['depth_input'] = args['depth_input']
    config['max_episode_len'] = args['max_episode_len']
    return config


def train(args=None, warmstart=None):

    # Process Observation Shape
    common.process_observation_shape(model='rnn',
                                     resolution_level=args['resolution_level'],
                                     segmentation_input=args['segment_input'],
                                     depth_input=args['depth_input'],
                                     history_frame_len=1)

    args['logger'] = utils.MyLogger(args['log_dir'], True)
    trainer = create_zmq_trainer(args['algo'], model='rnn', args=args)
    if warmstart is not None:
        if os.path.exists(warmstart):
            print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            save_dir = args['save_dir']
            print('Warmstarting from save_dir <{}> with version <{}> ...'.format(save_dir, warmstart))
            trainer.load(save_dir, warmstart)

    name = 'ipc://whatever'
    name2 = 'ipc://whatever2'

    n_proc = args['n_proc']
    config = create_zmq_config(args)
    procs = [ZMQSimulator(k, name, name2, config) for k in range(n_proc)]
    [k.start() for k in procs]
    ensure_proc_terminate(procs)

    master = ZMQMaster(name, name2, trainer=trainer, config=args)

    # both loops must be running
    print('Start Iterations ....')
    send_thread = threading.Thread(target=master.send_loop, daemon=True)
    send_thread.start()
    master.recv_loop()
    print('Done!')
    trainer.save(args['save_dir'], version='final')


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for 3D House Navigation")
    # Environment
    parser.add_argument("--n-house", type=int, default=1,
                        help="number of houses to train on. Should be no smaller than --n-proc")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    #parser.add_argument("--linear-reward", action='store_true', default=False,   # by default linear reward
    #                    help="whether to use reward according to distance; o.w. indicator reward")
    #parser.add_argument("--action-dim", type=int, help="degree of freedom of agent movement, must be in the range of [2, 4], default=4")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none', dest='segment_input',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'],
                        dest='resolution_level', default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    #parser.add_argument("--history-frame-len", type=int, default=4,
    #                    help="length of the stacked frames, default=4")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")

    ########################################################
    # ZMQ training parameters
    parser.add_argument("--train-gpu", type=int,
                        help="[ZMQ] an integer indicating the training gpu")
    parser.add_argument("--render-gpu", type=str,
                        help="[ZMQ] an integer or a ','-split list of integers, indicating the gpu-id for renderers")
    parser.add_argument("--n-proc", type=int, default=32,
                        help="[ZMQ] number of processes for simulation")
    parser.add_argument("--t-max", type=int, default=5,
                        help="[ZMQ] number of time steps in each batch")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[ZMQ] batch size, should be no greather than --num-proc")

    ###########################################################
    # Core training parameters
    parser.add_argument("--algo", choices=['a3c'], default="a3c", help="algorithm")
    parser.add_argument("--lrate", type=float, help="learning rate for policy")
    parser.add_argument('--weight-decay', type=float, help="weight decay for policy")
    parser.add_argument("--gamma", type=float, help="discount")
    parser.add_argument("--grad-clip", type=float, default = 5.0, help="gradient clipping")
    parser.add_argument("--max-iters", type=int, default=int(1e6), help="maximum number of training episodes")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam', help="optimizer")
    parser.add_argument("--exploration-scheduler", choices=['low', 'medium', 'high', 'none', 'linear', 'exp'],
                        dest='scheduler', default='none',
                        help="Whether to use eps-greedy scheduler to execute exploration. Default none")

    ####################################################
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int, default=256,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int, default=1,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'], default='lstm',
                        help="[RNN-Only] RNN cell type")

    ####################################################
    # Aux Tasks and Additional Sampling Choice
    parser.add_argument("--q-loss-coef", type=float,
                        help="For joint model, the coefficient for q_loss")

    ###################################################
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many training iters are completed")
    parser.add_argument("--report-rate", type=int, default=1,
                        help="report training stats once every time this many training steps are performed")
    parser.add_argument("--eval-rate", type=int, default=50,
                        help="report evaluation stats once every time this many *FRAMES* produced")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)  #optional

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    args = cmd_args.__dict__
    args['scheduler'] = create_scheduler(cmd_args.scheduler)

    train(args, warmstart=cmd_args.warmstart)
