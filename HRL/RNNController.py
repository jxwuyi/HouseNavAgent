from headers import *
import common
import utils

import sys, os, platform, time

import numpy as np
import random

import House3D
from House3D.roomnav import n_discrete_actions
from House3D import Environment as HouseEnv
from House3D import MultiHouseEnv
from House3D import House
from House3D.house import ALLOWED_PREDICTION_ROOM_TYPES, ALLOWED_OBJECT_TARGET_INDEX, ALLOWED_TARGET_ROOM_TYPES, ALLOWED_OBJECT_TARGET_TYPES
from House3D.roomnav import RoomNavTask
from House3D.objnav import ObjNavTask

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


"""
Learning a RNNPlanner over Objects and Rooms
Rooms: 8 rooms total
Objects: 15 objects total

Observation-Shape: Dim = 23 * 4 = 92
  > mask_feture: 23
  > accumulative mask feature: 23
  > target: 23
  > last_action: 23
"""

combined_target_list = ALLOWED_TARGET_ROOM_TYPES + ALLOWED_OBJECT_TARGET_TYPES
combined_target_index = dict()
for i, t in enumerate(combined_target_list):
    combined_target_index[t] = i

n_mask_feature = len(ALLOWED_OBJECT_TARGET_TYPES) + len(ALLOWED_TARGET_ROOM_TYPES)   # 23

n_rooms = len(ALLOWED_TARGET_ROOM_TYPES)
n_objects = len(ALLOWED_OBJECT_TARGET_TYPES)


def _target_to_one_hot(target_id):
    ret = np.zeros(n_mask_feature, dtype=np.uint8)
    if target_id > -1: ret[target_id] = 1
    return ret


def _feature_mask_to_names(mask):
    return [_t for _i, _t in enumerate(combined_target_list) if mask[_i] > 0]


def _feature_mask_to_index(mask):
    return [_i for _i in range(n_mask_feature) if mask[_i] > 0]


def _get_room_index_from_mask(mask):
    if np.sum(mask[:n_rooms-1]) == 0:
        return [n_rooms-1]  # only indoor
    return [_i for _i in range(n_rooms - 1) if mask[_i] > 0]


def _get_object_index_from_mask(mask):
    base = n_rooms - 1
    return [_i for _i in range(n_objects) if mask[base + _i] > 0]


def _mask_feature_to_bits(mask):
    ret = 0
    for i in range(n_mask_feature):
        if mask[i] > 0:
            ret |= 1 << i
    return ret


def _log_it(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger.print(msg)


def load_policy(policy, filename):
    if os.path.exists(filename):
        policy.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    else:
        print('[Warning!!] policy model file not found! loading skipped... filename = <{}>'.format(filename))


def save_policy(policy, save_dir, name="RNNPlanner", version=""):
    if len(version) > 0:
        version = "_" + version
    if save_dir[-1] != '/':
        save_dir += '/'
    try:
        filename = save_dir + name + version + '.pkl'
        torch.save(policy.state_dict(), filename)
    except Exception as e:
        print('[RNNController.save] fail to save model <{}>! Err = {}... Saving Skipped ...'.format(filename, e), file=sys.stderr)


class SimpleRNNPolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super(SimpleRNNPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim   # action space
        self.hidden_units = hidden_units

        # build rnn
        self.cell = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_units,
                            num_layers=1,
                            batch_first=True)
        utils.initialize_weights(self.cell)

        # build policy layers
        self.policy_layer = nn.Linear(self.hidden_units, self.output_dim)
        utils.initialize_weights(self.policy_layer, True)

        # build critic layers
        self.critic_layer = nn.Linear(self.hidden_units, 1)
        utils.initialize_weights(self.critic_layer)

        # internal hidden
        self._last_h = None

    def get_zero_state(self, batch=1, return_variable=True, volatile=False):
        z = torch.zeros(1, batch, self.hidden_units).type(FloatTensor)
        if return_variable: z = Variable(z, volatile=volatile)
        return (z, z)

    def forward(self, x, h, sample_action=False, return_tensor=False):
        """
        compute the forward pass of the model.
        @:param x: [batch, seq_len, input_dim]
        @:param h: hidden state, LSTM, ([1, batch, hidden], [1, batch, hidden])
        @:param only_action: when True, only return a sampled action, LongTensor, [batch, seq_len, 1]
        @:param return_tensor: when True, return tensor
        @:return (logp, value, next_hidden) or (sampled_action, nxt_hidden)
        """

        seq_len = x.size(1)
        batch_size = x.size(0)

        output, nxt_h = self.cell(x, h)   # output: [batch, seq_len, hidden_units]
        if return_tensor:
            nxt_h = (nxt_h[0].data, nxt_h[1].data)

        # compute action
        feat = output.view(-1, self.hidden_units)
        flat_logits = self.policy_layer(feat)
        self.logits = flat_logits.view(batch_size, seq_len, self.output_dim)
        if sample_action:
            flag_prob = F.softmax(flat_logits)
            self.prob = flag_prob.view(batch_size, seq_len, self.output_dim)
            act = torch.multinomial(flag_prob, 1).view(batch_size, seq_len, 1)
            if return_tensor:
                act = act.data
            return act, nxt_h

        self.logp = ret_logp = F.log_softmax(flat_logits).view(batch_size, seq_len, self.output_dim)
        if return_tensor:
            ret_logp = ret_logp.data

        # compute value
        flat_value = self.critic_layer(feat)
        self.value = ret_value = flat_value.view(batch_size, seq_len)
        if return_tensor:
            ret_value = ret_value.data

        return ret_logp, ret_value, nxt_h

    def entropy(self, logits=None):
        """
        logits: [batch, seq_len, D_out]
        return: [batch, seq_len]
        """
        if logits is None: logits = self.logits
        a0 = logits - logits.max(dim=2, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = ea0.sum(dim=2, keepdim=True)
        p0 = ea0 / z0
        ret = p0 * (torch.log(z0 + 1e-8) - a0)
        return ret.sum(dim=2)



class RNNPlanner(object):
    def __init__(self, motion, rnn_units=50, warmstart=None):
        self.task = motion.task
        self.env = self.task.env
        self.motion = motion
        self.n_obs_dim = n_mask_feature * 4  # feature, accu_feature, target, last_option
        self.n_act_dim = n_mask_feature
        self.rnn_units = rnn_units
        self.policy = SimpleRNNPolicy(self.n_obs_dim, self.n_act_dim, self.rnn_units)
        if use_cuda:
            self.policy.cuda()
        if warmstart is not None:
            print('[RNNPlanner] Loading Planner Policy ...')
            load_policy(self.policy, warmstart)

    def save(self, save_dir, version=""):
        save_policy(self.policy, save_dir, version=version)

    """
    n_iters: training iterations
    motion_steps: maximum steps of motion execution
    planner_step: maximum of planner steps
    """
    def learn(self, n_iters=20000, episode_len=300,
              motion_steps=50, planner_step=10, batch_size=64,
              lrate=0.001, weight_decay=0.00001, entropy_penalty=0.01,
              gamma=0.99, time_penalty=0.1, success_reward=2,
              grad_clip=1,
              logger=None, seed=None):
        ts = time.time()
        if seed is not None:
            np.random.seed(seed)
        _log_it(logger, "Training RNN Planner...")
        self.optim = optim.Adam(self.policy.parameters(), lr=lrate, weight_decay=weight_decay)
        total_episodes = 0
        episode_stats = dict(good=[], meters=[], target=[], steps=[])
        for _iter in range(n_iters):
            _log_it(logger, "Start Iteration#{} ...".format(_iter))
            # collecting
            _log_it(logger, ">> Collecting Samples ...")
            data = []
            tt = time.time()

            total_samples = 0
            for _ep in range(batch_size):
                cur_obs = []
                cur_info = []
                self.task.reset()
                final_target_name = task.get_current_target()
                final_target_id = combined_target_index[final_target_name]
                final_target = _target_to_one_hot(final_target_id)
                accu_mask = _target_to_one_hot(-1)
                last_feature = self.task.get_feature_mask()
                last_option = -1
                last_h = self.policy.get_zero_state()
                flag_done = False
                # store episode stats
                episode_stats['meters'].append(self.task.info['meters'])
                episode_stats['target'].append(final_target_name)
                ep_steps = 0
                # episode
                for _step in range(planner_step):
                    planner_input_np = np.concatenate([last_feature,
                                                       accu_mask,
                                                       _target_to_one_hot(last_option),
                                                       final_target])
                    planner_input = torch.from_numpy(planner_input_np).type(FloatTensor)
                    cur_obs.append(planner_input)  # store data
                    # get current option
                    act_tf = self.policy(Variable(planner_input.view(1, 1, -1)), last_h, sample_action=True, return_tensor=True)  # [batch, seq]
                    last_option = act_tf.numpy().flatten()[0]
                    total_samples += 1
                    # run locomotion
                    motion_data = self.motion.run(combined_target_list[last_option], motion_steps)
                    # process data
                    accu_mask = last_feature
                    for dat in motion_data[:-1]:
                        accu_mask |= dat[0]
                        ep_steps += 1
                        if dat[0][final_target_id] > 0:
                            flag_done = True
                            last_feature = dat[0]
                            break
                    cur_info.append((last_option, flag_done, flag_done or (motion_data[-1][last_option] > 0)))
                    if flag_done: break
                    ep_steps += 1
                    last_feature = motion_data[-1][0]
                # update epsiode stats
                episode_stats['steps'].append(ep_steps)
                episode_stats['good'].append(1 if flag_done else 0)
                # extra frame for actor-critic
                planner_input_np = np.concatenate([last_feature, accu_mask, _target_to_one_hot(last_option), final_target])
                cur_obs.append(torch.from_numpy(planner_input_np).type(FloatTensor))
                data.append((cur_obs, cur_info))
            print('  --> Done! Total <{}> Samples Collected! Time Elapsed = %.4fs' % (time.time() - tt))
            total_episodes += batch_size

            # Perform Training
            # TODO

        dur = time.time() - ts
        _log_it(logger, ("Training Done! Total Computation Time = %.4fs" % dur))

    def observe(self, exp_data, target):
        raise NotImplementedError()

    def plan(self, mask, target, return_list = False):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
