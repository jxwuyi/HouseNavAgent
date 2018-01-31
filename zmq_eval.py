from headers import *
import common
import utils

import environment
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

class ZMQHouseEnvironment:
    def __init__(self, k=0, reward_type='indicator', success_measure='center', multi_target=False, aux_task=False,
                 hardness=None, segment_input='none', depth_input=False, max_steps=-1, device=0, seed=0):
        assert k >= 0
        np.random.seed(seed)
        random.seed(seed)
        self.env = common.create_env(k, reward_type=reward_type, hardness=hardness, success_measure=success_measure,
                                     segment_input=segment_input, depth_input=depth_input,
                                     max_steps=max_steps, render_device=device,
                                     genRoomTypeMap=aux_task,
                                     cacheAllTarget=multi_target)
        self.obs = self.env.reset()
        self.done = False
        self.multi_target = multi_target
        if multi_target:
            self.env.cache_all_target()
        self.aux_task = aux_task
        if self.aux_task:
            self._aux_target = self.env.get_current_room_pred_mask()
        self._target = common.target_instruction_dict[self.env.get_current_target()]

    def current_state(self):
        if self.aux_task:
            return self.obs, self._target, self._aux_target
        else:
            return self.obs, self._target

    def action(self, act):
        if act is None:
            exit(0)
        obs, rew, done, _ = self.env.step(act, return_info=False)
        if done:
            if self.multi_target:
                obs = self.env.reset(reset_target=True)
                self._target = common.target_instruction_dict[self.env.get_current_target()]
            else:
                obs = self.env.reset()
        self.obs = obs
        return rew, done

class ZMQSimulator(SimulatorProcess):
    def _build_player(self):
        config = self.config
        k = self.idx % config['n_house']
        # set random seed
        np.random.seed(self.idx)
        device_list = config['render_devices']
        device_ind = self.idx % len(device_list)
        device = device_list[device_ind]
        return ZMQHouseEnvironment(k, config['reward_type'], config['success_measure'],
                                   config['multi_target'], config['aux_task'], config['hardness'],
                                   config['segment_input'], config['depth_input'],
                                   config['max_episode_len'], device, seed=self.idx)


class ZMQMaster(SimulatorMaster):
    def __init__(self, pipe1, pipe2, trainer, config):
        super(ZMQMaster, self).__init__(pipe1, pipe2)
        self.config = config
        self.logger = config['logger']
        self.cnt = 0
        self.n_action = common.n_discrete_actions  # TODO: to allow further modification
        self.train_buffer = dict()
        self.pool = set()
        self.hidden_state = dict()
        self.accu_stats = dict()
        self.epis_cnt = dict()
        self.start_time = time.time()
        self.episode_stats = dict(len=[], rew=[], succ=[])
        self.multi_target = config['multi_target']
        self.max_iters = config['max_iters']
        self.n_house = config['n_house']
        assert self.max_iters % self.n_house == 0
        self.avg_iters = self.max_iters / self.n_house
        if self.multi_target:
            self.episode_stats['target'] = []
            self.curr_target = dict()
        self.aux_task = config['aux_task']
        if self.aux_task:
            self.episode_stats['aux_task_rew'] = []
            self.episode_stats['aux_task_err'] = []
        self.trainer = trainer

    def save_all(self, version=''):
        self.logger.print('>>> Saving to log_dir = <{}> ...'.format(self.config['log_dir']))
        self.trainer.save(self.config['log_dir'], version=version+'_epis_stats', target_dict_data=self.episode_stats)

    def recv_message(self, ident, _state, reward, isOver):
        """
        Handle a message sent by "ident" simulator.
        The simulator takes the last action we send, arrive in "state", get "reward" and "isOver".
        """
        if self.aux_task:
            state, target, aux_msk = _state
        else:
            state, target = _state
            aux_msk = None
        trainer = self.trainer
        if ident not in self.hidden_state:  # new process passed in
            self.epis_cnt[ident] = 0
            if trainer.is_rnn():
                self.hidden_state[ident] = trainer.get_init_hidden()
            self.accu_stats[ident] = dict(rew=0, len=0, succ=0)
            if self.multi_target:
                self.accu_stats[ident]['target'] = common.all_target_instructions[target]
            if self.aux_task:
                self.accu_stats[ident]['aux_task_rew'] = 0
                self.accu_stats[ident]['aux_task_err'] = 0

        self.accu_stats[ident]['rew'] += reward
        self.accu_stats[ident]['len'] += 1

        if isOver:
            # clear hidden state
            if isinstance(self.hidden_state[ident], tuple):
                c, g = self.hidden_state[ident]
                c *= 0.0
                g *= 0.0
                self.hidden_state[ident] = (c, g)
            else:
                self.hidden_state[ident] *= 0.0
            # accumulate running stats
            if reward > 5:  # magic number, since when we succeed we have a super large reward
                self.accu_stats[ident]['succ'] = 1
            self.episode_stats['rew'].append(self.accu_stats[ident]['rew'])
            self.episode_stats['len'].append(self.accu_stats[ident]['len'])
            self.episode_stats['succ'].append(self.accu_stats[ident]['succ'])
            self.accu_stats[ident]['rew'] = 0
            self.accu_stats[ident]['len'] = 1
            self.accu_stats[ident]['succ'] = 0
            self.epis_cnt[ident] += 1
            if self.multi_target:
                self.episode_stats['target'].append(self.accu_stats[ident]['target'])
                self.accu_stats[ident]['target'] = target
            if self.aux_task:
                self.episode_stats['aux_task_err'].append(self.accu_stats[ident]['aux_task_err'] / self.episode_stats['len'][-1])
                self.accu_stats[ident]['aux_task_err'] = 0
                self.episode_stats['aux_task_rew'].append(self.accu_stats[ident]['aux_task_rew'])
                self.accu_stats[ident]['aux_task_rew'] = 0
            if len(self.episode_stats['succ']) >= self.max_iters:
                print('>>>>> Done!!!!')
                self.logger.print('>>>>> DONE!!!!')
                self.logger.print('####### Final Stats #######')
                self.logger.print(">>> Succ Rate = %.3f" % (float(np.sum(self.episode_stats['succ'])) / self.max_iters))
                self.logger.print(">>> Avg Succ Path Len = %.3f" % (float(np.mean([l for s,l in zip(self.episode_stats['succ'], self.episode_stats['len']) if s > 0]))))
                if self.multi_target:
                    all_targets = sorted(list(set(self.episode_stats['target'])))
                    for tar in all_targets:
                        eps = [(s, l) for t, s, l in zip(self.episode_stats['target'], self.episode_stats['succ'], self.episode_stats['len']) if t == tar]
                        self.logger.print('  --> Target = %s, Rate = %.3f, Succ = %.3f, AvgLen = %.3f' % (tar, len(eps) / self.max_iters, float(np.mean([e[0] for e in eps])), float(np.mean([e[1] for e in eps if e[0] > 0])) ))
                self.save_all()
                exit(0)
            if self.epis_cnt[ident] >= self.avg_iters:
                return None  # stop the process!

        if isinstance(state, np.ndarray): state = torch.from_numpy(state).type(ByteTensor)
        self.curr_state[ident] = state
        self.curr_target[ident] = target
        if self.aux_task:
            self.curr_aux_mask[ident] = aux_msk

        # currently run batched simulation
        if ident in self.train_buffer:
            self.train_buffer[ident]['obs'].append(state)
            self.train_buffer[ident]['done'].append(isOver)
            self.train_buffer[ident]['rew'].append(reward)
            if self.multi_target: self.train_buffer[ident]['target'].append(target)
            if self.aux_task: self.train_buffer[ident]['aux_target'].append(trainer.process_aux_target(aux_msk))
            self.pool.add(ident)
            if len(self.pool) == self.batch_size:
                if self.batch_step == self.t_max:
                    self._perform_train()
                    self.train_buffer.clear()
                    self.batch_step = 0
                else:
                    self.batch_step += 1
                    self._batched_simulate()
                self.pool.clear()

        # no batch selected, create a batch and initialize the first action
        if len(self.train_buffer) == 0:
            if len(self.hidden_state) >= self.batch_size:
                cand = self._rand_select(self.hidden_state.keys())
                for id in cand:
                    self.train_buffer[id] = dict(obs=[self.curr_state[id]], rew=[], done=[], act=[],
                                                 init_h=self.hidden_state[id], target=[self.curr_target[id]])
                    if self.aux_task:
                        self.train_buffer[id]['aux_target'] = [trainer.process_aux_target(self.curr_aux_mask[id])]
                self._batched_simulate()
                self.pool.clear()
                self.batch_step += 1

        # report stats
        self.comm_cnt += 1
        if self.comm_cnt % self.config['eval_rate'] == 0:
            self._evaluate_stats()


def create_zmq_config(args):
    config = dict()

    # env param
    config['n_house'] = args['n_house']
    config['reward_type'] = args['reward_type']
    config['hardness'] = args['hardness']
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
    config['max_episode_len'] = args['max_episode_len']
    config['success_measure'] = args['success_measure']
    config['multi_target'] = args['multi_target']
    config['aux_task'] = args['aux_task']
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

    pipedir = os.environ.get('ZMQ_PIPEDIR', '.')
    name = 'ipc://{}/whatever'.format(pipedir)
    name2 = 'ipc://{}/whatever2'.format(pipedir)
    n_proc = args['n_proc']
    config = create_zmq_config(args)
    procs = [ZMQSimulator(k, name, name2, config) for k in range(n_proc)]
    [k.start() for k in procs]
    ensure_proc_terminate(procs)

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
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test'], default='small')
    parser.add_argument("--n-house", type=int, default=1,
                        help="number of houses to train on. Should be no larger than --n-proc")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--linear-reward", action='store_true', default=False,
                        help="[Deprecated] whether to use reward according to distance; o.w. indicator reward")
    parser.add_argument("--reward-type", choices=['none', 'linear', 'indicator', 'delta', 'speed'], default='indicator',
                        help="Reward shaping type")
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
    parser.add_argument("--success-measure", choices=['center', 'stay', 'see'], default='center',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
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
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    common.set_house_IDs(cmd_args.env_set, ensure_kitchen=(not cmd_args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (cmd_args.env_set, len(common.all_houseIDs)))

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
    args = cmd_args.__dict__

    args['model_name'] = 'rnn'
    args['scheduler'] = create_scheduler(cmd_args.scheduler)

    train(args, warmstart=cmd_args.warmstart)
