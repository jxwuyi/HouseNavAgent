from headers import *
import common
import utils

import environment
import elf_multihouse_env
from trainer.elf_ddpg_joint import ELF_JointDDPGTrainer as JointTrainer
from trainer.elf_a3c import ELF_A3CTrainer as A3CTrainer

from elf_python import GCWrapper

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_scheduler(type='medium'):
    if type == 'none':
        return utils.ConstantSchedule(1.0)
    if type == 'linear':
        return utils.LinearSchedule(10000, 1.0, 0.0)
    if type == 'medium':
        endpoints = [(0, 0), (2000, 0.1), (5000, 0.25), (10000, 0.5), (20000, 1.0)]
    elif type == 'high':
        endpoints = [(0, 0), (3000, 0.1), (8000, 0.25), (15000, 0.5), (30000, 1.0)]
    elif type == 'low': # low
        endpoints = [(0, 0), (1000, 0.1), (2000, 0.25), (7000, 0.5), (15000, 1.0)]
    elif type == 'exp':
        endpoints = [(0, 0), (1000, 0.01), (5000, 0.1), (8000, 0.5), (10000, 0.75), (12000, 0.9), (20000, 0.95), (30000, 1.0)]
    print('Building PiecewiseScheduler with <endpoints> = {}'.format(endpoints))
    scheduler = utils.PiecewiseSchedule(endpoints, outside_value=1.0)
    return scheduler


def create_elf_trainer(algo, model, args):
    observation_shape = common.observation_shape
    action_shape = common.action_shape
    if algo == 'ddpg':
        model_gen = lambda: common.create_joint_model(args, observation_shape, action_shape)
        trainer = JointTrainer('JointDDPGTrainer', model_gen,
                               observation_shape, action_shape, args)
    elif algo == 'a3c':
        model_gen = lambda: common.create_discrete_model(algo, args, observation_shape)
        trainer = A3CTrainer('A2CTrainer', model_gen, observation_shape,
                             environment.n_discrete_actions, args)
    else:
        assert False, '[ELF Trainer] Trainer <{}> is not defined!'.format(algo)
    return trainer


def create_elf_env(houseID, linearReward, hardness, args):
    worlds = common.create_world_from_index(houseID)
    segment_input = args['segment_input']
    act_dim = 1 if args['algo'] == 'a3c' else sum(common.action_shape)
    gpu_ids = args['render_gpu'].split(',')
    render_gpus = [int(k) for k in gpu_ids]

    config = dict(
        worlds=worlds,
        group_size=args['env_group_size'],
        render_gpus=render_gpus,
        seed=args['seed'],
        color_file=common.colorFile,
        resolution=common.resolution,
        linear_reward=linearReward,
        hardness=hardness,
        action_degree=common.action_shape[0],
        use_segment_input=(segment_input != 'none'),
        use_segment_id=(segment_input == 'index'),
        use_joint_signal=(segment_input == 'joint'),
        history_frames=common.frame_history_len,
        epsiode_length=args['episode_len'],
        act_dim=act_dim
    )

    desc = dict(
        actor=dict(
            input=dict(s=""),
            reply=dict(a=""),
            connector="actor-connector"
        ),
        train=dict(
            input=dict(s="", last_r="", last_next="", last_a="", last_done="", stats_eplen="", stats_rew=""),
            reply=None,
            connector="trainer-connector"
        )
    )
    wrapper = GCWrapper(elf_multihouse_env.ELF_MultiHouseEnv, desc, args['num_games'], args['batch_size'], args['elf_T'], config=config)
    return wrapper


def train(args=None, seed=None,
          houseID=0, linearReward=False, algo='pg',
          model_name='cnn',  # NOTE: optional: model_name='rnn'
          iters=2000000, eval_range=200,
          log_dir='./log', save_dir='./_model_', warmstart=None,
          log_debug_info=True):
    """
    >>>> Do Not Use Scheduler Now
    if 'scheduler' in args:
        scheduler = args['scheduler']
    else:
        scheduler = None
    """

    if args is None:
        args = common.create_default_args(algo)
    args['algo'] = algo

    hardness = args['hardness']
    if hardness is not None:
        print('>>> Hardness Level = {}'.format(hardness))

    trainer = create_elf_trainer(algo, model_name, args)
    wrapper = create_elf_env(houseID, linearReward, hardness, args)

    logger = args['logger']

    if warmstart is not None:
        if os.path.exists(warmstart):
            logger.print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            logger.print('Warmstarting from save_dir <{}> with version <{}> ...'.format(save_dir, warmstart))
            trainer.load(save_dir, warmstart)

    logger.print('Start Training')

    if log_debug_info:
        common.debugger = utils.MyLogger(log_dir, True, 'full_logs.txt')
    else:
        common.debugger = utils.FakeLogger()

    wrapper.reg_callback("train", trainer.update)
    wrapper.reg_callback("actor", trainer.actor)

    print('Starting iterations...')

    for _ in range(iters):
        wrapper.Run()

    print('>> Done!')


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for 3D House Navigation")
    # Environment
    parser.add_argument("--house", type=int, default=0,
                        help="house ID (default 0); if < 0, then multi-house environment")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--linear-reward", action='store_true', default=False,
                        help="whether to use reward according to distance; o.w. indicator reward")
    parser.add_argument("--action-dim", type=int, help="degree of freedom of agent movement, must be in the range of [2, 4], default=4")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'], default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--history-frame-len", type=int, default=4,
                        help="length of the stacked frames, default=4")
    parser.add_argument("--max-episode-len", type=int, help="maximum episode length")
    # ELF parameters
    parser.add_argument("--render-gpu", type=str, default="0",
                        help="[ELF] an integer or a ','-split list of integers, indicating the gpu-id for renderers")
    parser.add_argument("--num-games", type=int, default=64,
                        help="[ELF] number of threads for simulating environments in elf (default=20)")
    parser.add_argument("--env-group-size", type=int, default=10,
                        help="[ELF] number of threads to split all the houses (default=10)")
    parser.add_argument("--elf-T", type=int, default=5,
                        help="[ELF] number of time steps to run in each thread")  # last time step will be dropped
    parser.add_argument("--batch-size", type=int, default=32, help="[ELF] batch size in elf; True <batchsize> is [elf-T] * [batch-size]")
    # Core training parameters
    parser.add_argument("--algo", choices=['ddpg','a3c'], default="ddpg", help="algorithm")
    parser.add_argument("--lrate", type=float, help="learning rate for policy")
    parser.add_argument('--weight-decay', type=float, help="weight decay for policy")
    parser.add_argument("--gamma", type=float, help="discount")
    parser.add_argument("--max-iters", type=int, default=int(2e6), help="maximum number of training episodes")
    parser.add_argument("--target-net-update-rate", type=float, help="update rate for target networks")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--critic-penalty", type=float, default=0.001, help="critic norm regularizer")
    parser.add_argument("--noise-scheduler", choices=['low','medium','high','none','linear','exp'],
                        dest='scheduler', default='medium',
                        help="Whether to use noise-level scheduler to control the smoothness of action output. default=False.")
    parser.add_argument("--use-action-gating", dest='action_gating', action='store_true',
                        help="whether to use action gating structure in the critic model")
    parser.set_defaults(action_gating=False)
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--batch-length", type=int,
                        help="[RNN-Only] maximum length of an episode in a batch")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'],
                        help="[RNN-Only] RNN cell type")
    # Aux Tasks and Additional Sampling Choice
    parser.add_argument("--dist-sampling", dest='dist_sample', action="store_true")
    parser.set_defaults(dist_sample=False)
    parser.add_argument("--q-loss-coef", type=float,
                        help="For joint model, the coefficient for q_loss")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--report-rate", type=int, default=50, help="report training stats once every time this many training steps are performed")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")
    parser.add_argument("--debug", action="store_true", dest="debug", help="log all the computation details")
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="turn off debug logs")
    parser.set_defaults(debug=False)
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)  #optional

    if cmd_args.action_dim is not None:
        print('Degree of freedom set to be <{}>!'.format(cmd_args.action_dim))
        common.action_shape = (cmd_args.action_dim, 2)

    if cmd_args.linear_reward:
        print('Using Linear Reward Function in the Env!')

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    args = common.create_default_args(cmd_args.algo, cmd_args.gamma,
                               cmd_args.lrate, None,
                               cmd_args.max_episode_len, cmd_args.batch_size,
                               None,
                               cmd_args.use_batch_norm,
                               cmd_args.entropy_penalty,
                               cmd_args.critic_penalty,
                               cmd_args.weight_decay,
                               None,
                               None,
                               # RNN Parameters
                               cmd_args.batch_length, cmd_args.rnn_layers,
                               cmd_args.rnn_cell, cmd_args.rnn_units,
                               # input type
                               cmd_args.segmentation_input,
                               cmd_args.resolution,
                               cmd_args.history_frame_len)

    if cmd_args.target_net_update_rate is not None:
        args['target_net_update_rate']=cmd_args.target_net_update_rate

    if cmd_args.hardness is not None:
        args['hardness'] = cmd_args.hardness

    args['scheduler'] = create_scheduler(cmd_args.scheduler or 'none')

    if cmd_args.dist_sample:
        args['dist_sample'] = True

    if cmd_args.q_loss_coef is not None:
        args['q_loss_coef'] = cmd_args.q_loss_coef

    args['action_gating'] = cmd_args.action_gating   # gating in ddpg network

    args['seed'] = cmd_args.seed

    args['num_games'] = cmd_args.num_games
    args['env_group_size'] = cmd_args.env_group_size
    args['elf_T'] = cmd_args.elf_T
    args['render_gpu'] = cmd_args.render_gpu

    args['report_gap'] = cmd_args.report_rate
    args['save_rate'] = cmd_args.save_rate
    args['save_dir'] = cmd_args.save_dir
    args['logger'] = utils.MyLogger(cmd_args.log_dir, True)

    train(args,
          houseID=cmd_args.house, linearReward=cmd_args.linear_reward,
          algo=cmd_args.algo, iters=cmd_args.max_iters,
          log_dir = cmd_args.log_dir,
          save_dir=cmd_args.save_dir,
          warmstart=cmd_args.warmstart,
          log_debug_info=cmd_args.debug)
