from headers import *
import common
import utils

import threading

from zmq_trainer.zmq_actor_critic import ZMQA3CTrainer
from zmq_trainer.zmq_aux_task import ZMQAuxTaskTrainer
from zmq_trainer.zmq_util import ZMQSimulator, ZMQMaster
from zmq_trainer.zmqsimulator import SimulatorProcess, SimulatorMaster, ensure_proc_terminate

from policy.rnn_discrete_actor_critic import DiscreteRNNPolicy

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_curriculum_schedule(curriculum):
    if curriculum is None: return None
    try:
        val = tuple(map(int, curriculum.split(',')))
    except Exception as e:
        print('[Curriculum-Schedule Parser] Invalid Curriculum Input Format! Please input 3 comman-seperated integers!')
        return None
    if (len(val) != 3) or (min(val) < 1):
        print('[Curriculum-Schedule Parser] Invalid Curriculum Input Format! Please input 3 comman-seperated integers!')
        return None
    return val

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
                              linear_hiddens=[256],
                              policy_hiddens=[128, 64],
                              critic_hiddens=[64, 32],
                              rnn_cell=args['rnn_cell'],
                              rnn_layers=args['rnn_layers'],
                              rnn_units=args['rnn_units'],
                              multi_target=args['multi_target'],
                              use_target_gating=args['target_gating'],
                              aux_prediction=(common.n_aux_predictions if args['aux_task'] else None),
                              no_skip_connect=(args['no_skip_connect'] if 'no_skip_connect' in args else False),
                              pure_feed_forward=(args['feed_forward'] if 'feed_forward' in args else False))
    if common.use_cuda:
        if 'train_gpu' in args:
            model.cuda(device_id=args['train_gpu'])  # TODO: Actually we only support training on gpu_id=0
        else:
            model.cuda()
    return model


def create_zmq_trainer(algo, model, args):
    assert model == 'rnn', 'currently only support rnn policy!'
    observation_shape = common.observation_shape
    n_action = common.n_discrete_actions
    if algo == 'a3c':
        model_gen = lambda: create_policy(model, args, observation_shape, n_action)
        if ('aux_task' in args) and args['aux_task']:
            assert False, '<aux_task> is not supported currently!'
            #trainer = ZMQAuxTaskTrainer('ZMQAuxTaskA3CTrainer', model_gen, observation_shape, [n_action], args)
        else:
            trainer = ZMQA3CTrainer('ZMQA3CTrainer', model_gen, observation_shape, [n_action], args)
    else:
        assert False, '[ZMQ Trainer] Trainer <{}> is not defined!'.format(algo)
    return trainer


def create_zmq_config(args):
    config = dict()

    # task name
    config['task_name'] = args['task_name']
    config['false_rate'] = args['false_rate']

    # env param
    config['n_house'] = args['n_house']
    config['reward_type'] = args['reward_type']
    config['reward_silence'] = args['reward_silence']
    config['hardness'] = args['hardness']
    config['max_birthplace_steps'] = args['max_birthplace_steps']
    config['min_birthplace_grids'] = args['min_birthplace_grids']
    config['curriculum_schedule'] = args['curriculum_schedule']
    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU for rendering!'
    if args['render_gpu'] is not None:
        gpu_ids = args['render_gpu'].split(',')
        render_gpus = [all_gpus[int(k)] for k in gpu_ids]
    elif args['train_gpu'] is not None:
        k = args['train_gpu']
        render_gpus = all_gpus[:k] + all_gpus[k+1:]
    else:
        if len(all_gpus) == 1:
            render_gpus = all_gpus
        else:
            render_gpus = all_gpus[1:]
    config['render_devices'] = tuple(render_gpus)
    config['segment_input'] = args['segment_input']
    config['depth_input'] = args['depth_input']
    config['target_mask_input'] = args['target_mask_input']
    config['max_episode_len'] = args['max_episode_len']
    config['success_measure'] = args['success_measure']
    config['multi_target'] = args['multi_target']
    config['object_target'] = args['object_target']
    config['fixed_target'] = args['fixed_target']
    config['aux_task'] = args['aux_task']
    config['cache_supervision'] = args['cache_supervision']
    config['outdoor_target'] = args['outdoor_target']
    return config


def train(args=None, warmstart=None):

    # Process Observation Shape
    common.process_observation_shape(model='rnn',
                                     resolution_level=args['resolution_level'],
                                     segmentation_input=args['segment_input'],
                                     depth_input=args['depth_input'],
                                     target_mask_input=args['target_mask_input'],
                                     history_frame_len=1)

    args['logger'] = utils.MyLogger(args['log_dir'], True, keep_file_handler=not args['append_file'])

    name = 'ipc://@whatever' + args['job_name']
    name2 = 'ipc://@whatever' + args['job_name'] + '2'
    n_proc = args['n_proc']
    config = create_zmq_config(args)
    procs = [ZMQSimulator(k, name, name2, config) for k in range(n_proc)]
    [k.start() for k in procs]
    ensure_proc_terminate(procs)

    trainer = create_zmq_trainer(args['algo'], model='rnn', args=args)
    if warmstart is not None:
        if os.path.exists(warmstart):
            print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            save_dir = args['save_dir']
            print('Warmstarting from save_dir <{}> with version <{}> ...'.format(save_dir, warmstart))
            trainer.load(save_dir, warmstart)


    master = ZMQMaster(name, name2, trainer=trainer, config=args)

    try:
        # both loops must be running
        print('Start Iterations ....')
        send_thread = threading.Thread(target=master.send_loop, daemon=True)
        send_thread.start()
        master.recv_loop()
        print('Done!')
        trainer.save(args['save_dir'], version='final')
    except KeyboardInterrupt:
        master.save_all(version='last')
        raise


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for 3D House Navigation")
    # Special Job Tag
    parser.add_argument("--job-name", type=str, default='')
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    parser.add_argument("--false-rate", type=float, default=0, help='The Rate of Impossible Targets')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--n-house", type=int, default=1,
                        help="number of houses to train on. Should be no larger than --n-proc")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
    parser.add_argument("--min-birthplace-grids", type=int, default=0,
                        help="int, the minimum grid distance of the birthplace towards target. Default 0, namely possible to born with gird_dist=0.")
    parser.add_argument("--curriculum-schedule", type=str,
                        help="in format of <a,b,c>, comma seperated 3 ints, the curriculum schedule. a: start birthsteps; b: brithstep increment; c: increment frequency")
    parser.add_argument("--linear-reward", action='store_true', default=False,
                        help="[Deprecated] whether to use reward according to distance; o.w. indicator reward")
    parser.add_argument("--reward-type", choices=['none', 'linear', 'indicator', 'delta', 'speed', 'new'], default='indicator',
                        help="Reward shaping type")
    parser.add_argument("--reward-silence", type=int, default=0,
                        help="When set, the first <--reward-silence> step of each episode will not have any reward signal except collision penalty")
    #parser.add_argument("--action-dim", type=int, help="degree of freedom of agent movement, must be in the range of [2, 4], default=4")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none', dest='segment_input',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--target-mask-input", dest='target_mask_input', action='store_true',
                        help="whether to include target mask 0/1 signal as part of the input signal")
    parser.set_defaults(target_mask_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'],
                        dest='resolution_level', default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    #parser.add_argument("--history-frame-len", type=int, default=4,
    #                    help="length of the stacked frames, default=4")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--success-measure", choices=['stop', 'stay', 'see'], default='see',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also a target. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--fixed-target", type=str, help="fixed training targets: candidate values room, object or any-room/object")
    parser.add_argument("--no-outdoor-target", dest='outdoor_target', action='store_false',
                        help="when this flag is set, we will exclude <outdoor> target")
    parser.set_defaults(outdoor_target=True)
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
    parser.add_argument("--grad-batch", type=int, default=1,
                        help="[ZMQ] the actual gradient descent batch-size will be <grad-batch> * <batch-size>")

    ###########################################################
    # Core training parameters
    parser.add_argument("--algo", choices=['a3c'], default="a3c", help="algorithm")
    parser.add_argument("--supervised-learning", dest='cache_supervision', action='store_true',
                        help="when set, use Dagger style supervised learning + RL fine-tuning (when close to target)")
    parser.set_defaults(cache_supervision=False)
    parser.add_argument("--lrate", type=float, help="learning rate for policy")
    parser.add_argument('--weight-decay', type=float, help="weight decay for policy")
    parser.add_argument("--gamma", type=float, help="discount")
    parser.add_argument("--grad-clip", type=float, default = 5.0, help="gradient clipping")
    parser.add_argument("--adv-norm", dest='adv_norm', action='store_true',
                        help="perform advantage normalization (per-minibatch, not the full gradient batch)")
    parser.set_defaults(adv_norm=False)
    parser.add_argument("--rew-clip", type=int, help="if set [r], clip reward to [-r, r]")
    parser.add_argument("--max-iters", type=int, default=int(1e6), help="maximum number of training episodes")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--logits-penalty", type=float, help="policy logits regularizer")
    parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam', help="optimizer")
    parser.add_argument("--exploration-scheduler", choices=['low', 'medium', 'high', 'none', 'linear', 'exp'],
                        dest='scheduler', default='none',
                        help="Whether to use eps-greedy scheduler to execute exploration. Default none")
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)

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
    parser.add_argument("--auxiliary-task", dest='aux_task', action='store_true',
                        help="Whether to perform auxiliary task of predicting room types")
    parser.set_defaults(aux_task=False)
    parser.add_argument("--use-reinforce-loss", dest='reinforce_loss', action='store_true',
                        help="When true, use reinforce loss to train the auxiliary task loss")
    parser.set_defaults(reinforce_loss=False)
    parser.add_argument("--aux-loss-coef", dest='aux_loss_coef', type=float, default=1.0,
                        help="Coefficient for the Auxiliary Task Loss. Only effect when --auxiliary-task")

    ####################################################
    # Ablation Test Options
    parser.add_argument("--no-skip-connect", dest='no_skip_connect', action='store_true',
                        help="[A3C-LSTM Only] no skip connect. only takes the output of rnn to compute action")
    parser.set_defaults(no_skip_connect=False)
    parser.add_argument("--feed-forward-a3c", dest='feed_forward', action='store_true',
                        help="[A3C-LSTM Only] skip rnn completely. essentially cnn-a3c")
    parser.set_defaults(feed_forward=False)

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

    ###################################################
    # Logging Option
    parser.add_argument("--append-file-handler", dest='append_file', action='store_true',
                        help="[Logging] When set, the logger will be close when a log message is output and reopen in the next time.")
    parser.set_defaults(append_file=False)
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    common.set_house_IDs(cmd_args.env_set, ensure_kitchen=(not cmd_args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (cmd_args.env_set, len(common.all_houseIDs)))

    common.ensure_object_targets(cmd_args.object_target)

    if cmd_args.fixed_target is not None:
        allowed_targets = list(common.target_instruction_dict.keys()) + ['any-room']
        if cmd_args.object_target:
            allowed_targets.append('any-object')
        assert cmd_args.fixed_target in allowed_targets, '--fixed-target specified an invalid target <{}>!'.format(cmd_args.fixed_target)
        if not ('any' in cmd_args.fixed_target):
            common.filter_house_IDs_by_target(cmd_args.fixed_target)
            print('[ZMQ_Train.py] Filter Houses By Fixed-Target to N=<{}> Houses...'.format(len(common.all_houseIDs)))

    if cmd_args.n_house > len(common.all_houseIDs):
        print('[ZMQ_Train.py] No enough houses! Reduce <n_house> to [{}].'.format(len(common.all_houseIDs)))
        cmd_args.n_house = len(common.all_houseIDs)

    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)  #optional

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    if cmd_args.linear_reward:
        print('--linearReward option is now *Deprecated*!!! Use --reward-type option instead! Now force <reward_type == \'linear\'>')
        cmd_args.reward_type = 'linear'

    if cmd_args.grad_batch < 1:
        print('--grad-batch option must be a positive integer! reset to default value <1>!')
        cmd_args.grad_batch = 1

    args = cmd_args.__dict__

    args['model_name'] = 'rnn'
    args['scheduler'] = create_scheduler(cmd_args.scheduler)
    args['curriculum_schedule'] = create_curriculum_schedule(cmd_args.curriculum_schedule)

    train(args, warmstart=cmd_args.warmstart)
