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

n_planner_input_feature = n_mask_feature * 4

n_rooms = len(ALLOWED_TARGET_ROOM_TYPES)
n_objects = len(ALLOWED_OBJECT_TARGET_TYPES)


def _target_to_one_hot(target_id):
    ret = np.zeros(n_mask_feature, dtype=np.uint8)
    if target_id > -1: ret[target_id] = 1
    return ret


"""
cur_info.append((last_option, flag_done, flag_done or (motion_data[-1][last_option] > 0)))
cur_obs.append(torch.from_numpy(planner_input_np).type(FloatTensor))
batch_data: [(cur_obs, cur_info)..]
cur_obs: a list of FloatTensor
cur_info: a list of (last_option, done, option_done)
>> return X, Act, Done, Mask
"""
def process_batch_data(batched_data, time_penalty, success_reward):
    batch_size = len(batched_data)
    seq_len = max([len(info) for obs, info in batched_data])
    # X: [batch, seq_len, feature_dim]
    X = torch.zeros((batch_size, seq_len + 1, n_planner_input_feature)).type(FloatTensor)
    Act = torch.zeros((batch_size, seq_len)).type(LongTensor)
    R = torch.zeros((batch_size, seq_len)).type(FloatTensor)
    Done = torch.zeros((batch_size, seq_len)).type(FloatTensor)
    Mask = torch.zeros((batch_size, seq_len)).type(FloatTensor)
    it = 0
    for obs, info in batched_data:
        n = len(info)
        X[it, :n+1, :] = torch.stack(obs, dim=0)
        for i, info_t in enumerate(info):
            opt, _, d = info_t
            Act[it, i] = int(opt)
        if info[-1][2]:
            Done[it, n - 1] = 1
            if n > 1:
                R[it, :n-1] = -time_penalty
            R[it, n-1] = success_reward
        else:
            R[it, :n] = -time_penalty
        Mask[it, :n] = 1
        it += 1
    return X, Act, R, Done, Mask


def _my_mean(lis):
    if len(lis) == 0:
        return 0
    return np.mean(lis)


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
        feat = output.contiguous().view(-1, self.hidden_units)
        flat_logits = self.policy_layer(feat)
        self.logits = ret_logits = flat_logits.view(batch_size, seq_len, self.output_dim)
        if sample_action:
            flag_prob = F.softmax(flat_logits)
            self.prob = flag_prob.view(batch_size, seq_len, self.output_dim)
            act = torch.multinomial(flag_prob, 1).view(batch_size, seq_len, 1)
            if return_tensor:
                act = act.data
            return act, nxt_h

        if return_tensor:
            ret_logits = ret_logits.data

        self.logp = ret_logp = F.log_softmax(flat_logits).view(batch_size, seq_len, self.output_dim)
        if return_tensor:
            ret_logp = ret_logp.data

        # compute value
        flat_value = self.critic_layer(feat)
        self.value = ret_value = flat_value.view(batch_size, seq_len)
        if return_tensor:
            ret_value = ret_value.data

        return ret_logp, ret_logits, ret_value, nxt_h

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
        self.n_obs_dim = n_planner_input_feature  # feature, accu_feature, target, last_option
        self.n_act_dim = n_mask_feature
        self.rnn_units = rnn_units
        self.policy = SimpleRNNPolicy(self.n_obs_dim, self.n_act_dim, self.rnn_units)
        if use_cuda:
            self.policy.cuda()
        if warmstart is not None:
            print('[RNNPlanner] Loading Planner Policy ...')
            load_policy(self.policy, warmstart)
        self.save_dir=None
        # execution parameters
        self.last_option = -1
        self.accu_mask = np.zeros(n_mask_feature, dtype=np.uint8)
        self.last_mask = None
        self.last_hidden = self._zero_hidden = self.policy.get_zero_state() # variable

    def save(self, save_dir, version=""):
        save_policy(self.policy, save_dir, version=version)

    def _perform_train(self, X, Act, Rew, Done, Mask, n_samples):
        # clear grad
        batch_size = Act.size(0)
        seq_len = Act.size(1)
        self.optim.zero_grad()
        # forward pass
        X = Variable(X)
        P, L, V, _ = self.policy(X, self.init_h)
        L = L[:, :seq_len, :]  # logits
        P = P[:, :seq_len, :]  # remove last one
        # compute accumulative Reward
        V_data = V.data
        cur_r = V_data[:, seq_len]
        V = V[:, :seq_len]
        V_data = V_data[:, :seq_len]
        R_list = []
        for t in range(seq_len - 1, -1, -1):
            cur_r = Rew[:, t] + self.gamma * Done[:, t] * cur_r
            R_list.append(cur_r)
        R_list.reverse()
        R = torch.stack(R_list, dim=1)
        # Advantage Normalization
        Adv = (R - V_data) * Mask  # advantage
        avg_val = Adv.sum() / n_samples
        Adv = (Adv - avg_val) * Mask  # reduce mean
        std_val = np.sqrt(torch.sum(Adv ** 2) / n_samples)  # standard dev
        Adv = Variable(Adv / max(std_val, 0.1))
        # critic loss
        R = Variable(R)
        Mask = Variable(Mask)
        critic_loss = torch.sum(Mask * (R - V) ** 2) / n_samples
        # policy gradient loss
        Act = Variable(Act)  # [batch_size, seq_len]
        Act = Act.unsqueeze(2)  # [batch_size, seq_len, 1]
        P_Act = torch.gather(P, 2, Act).squeeze(dim=2)  # [batch_size, seq_len]
        pg_loss = -torch.sum(P_Act * Adv * Mask) / n_samples
        # entropy bonus
        P_Ent = torch.sum(self.policy.entropy(L) * Mask) / n_samples
        pg_loss -= self.entropy_penalty * P_Ent
        # backprop
        loss = pg_loss + critic_loss
        loss.backward()
        L_norm = torch.sum(torch.sum(L**2, dim=-1) * Mask) / n_samples
        ret_dict = dict(pg_loss=pg_loss.data.cpu().numpy()[0],
                        policy_entropy=P_Ent.data.cpu().numpy()[0],
                        critic_loss=critic_loss.data.cpu().numpy()[0],
                        logits_norm=L_norm.data.cpu().numpy()[0])
        # gradient clip
        utils.clip_grad_norm(self.policy.parameters(), self.grad_clip)
        # apply SGD step
        self.optim.step()
        return ret_dict

    def _show_stats(self, episode_stats, eval_range=500):
        cur_stats = episode_stats[-eval_range:]
        succ_rate = _my_mean([stat['good'] for stat in cur_stats])
        avg_opt = _my_mean([len(stat['options']) for stat in cur_stats])
        avg_rew = _my_mean([stat['reward'] for stat in cur_stats])
        succ_avg_opt = _my_mean([len(stats['options']) for stats in cur_stats if stats['good'] > 0])
        succ_avg_steps = _my_mean([stats['steps'] for stats in cur_stats if stats['good'] > 0])
        succ_avg_meters = _my_mean([stats['meters'] for stats in cur_stats if stats['good'] > 0])
        _log_it(self.logger, '++++++++++++ Training Stats +++++++++++')
        _log_it(self.logger, "  > Succ Rate = %.4f" % succ_rate)
        _log_it(self.logger, "  > Avg. Reward = %.4f" % avg_rew)
        _log_it(self.logger, "  > Avg. Options = %.4f" % avg_opt)
        _log_it(self.logger, "  > Avg. Succ Steps = %.4f" % succ_avg_steps)
        _log_it(self.logger, "  > Avg. Succ Options = %.4f" % succ_avg_opt)
        _log_it(self.logger, "  > Avg. Succ Meters = %.4f" % succ_avg_meters)
        _log_it(self.logger, '+++++++++++++++++++++++++++++++++++++++')
        if succ_rate > self.best_succ_rate:
            _log_it(self.logger, " ----> Best Succ Model Stored!")
            self.save(self.save_dir, version='succ')
            self.best_succ_rate = succ_rate
        if avg_rew > self.best_reward:
            _log_it(self.logger, " ----> Best Reward Model Stored!")
            self.save(self.save_dir, version='reward')
            self.best_reward = avg_rew
        return dict(succ_rate=succ_rate, avg_rew=avg_rew, avg_opt=avg_opt,
                    succ_avg_opt=succ_avg_opt,succ_avg_steps=succ_avg_steps,succ_avg_meters=succ_avg_meters)


    """
    n_iters: training iterations
    motion_steps: maximum steps of motion execution
    planner_step: maximum of planner steps
    """
    def learn(self, n_iters=10000, episode_len=300, target=None,
              motion_steps=50, planner_steps=10, batch_size=64,
              lrate=0.001, weight_decay=0.00001, entropy_penalty=0.01,
              gamma=0.99, time_penalty=0.1, success_reward=2,
              grad_clip=1,
              logger=None, seed=None,
              report_rate=5, eval_rate=20, save_rate=100,
              save_dir=None):
        ts = time.time()
        self.best_succ_rate = 0
        self.best_reward = -10000
        self.save_dir=save_dir
        if seed is not None:
            np.random.seed(seed)
        self.logger = logger
        _log_it(logger, "Training RNN Planner...")
        self.optim = optim.Adam(self.policy.parameters(), lr=lrate, weight_decay=weight_decay)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.entropy_penalty = entropy_penalty
        self.init_h = self.policy.get_zero_state(batch_size, return_variable=True)
        # run iterations
        total_episodes = 0
        episode_stats = [] # dict(good=[], meters=[], target=[], steps=[], options=[], reward=[])
        train_stats = []
        eval_stats = []
        best_eval_rate = 0
        for _iter in range(n_iters):
            _log_it(logger, "Start Iteration#{} ...".format(_iter))
            # collecting
            _log_it(logger, "> Collecting Samples ...")
            data = []
            tt = time.time()

            total_samples = 0
            for _ep in range(batch_size):
                cur_obs = []
                cur_info = []
                self.task.reset(target=target)
                final_target_name = self.task.get_current_target()
                final_target_id = combined_target_index[final_target_name]
                final_target = _target_to_one_hot(final_target_id)
                accu_mask = _target_to_one_hot(-1)
                last_feature = self.task.get_feature_mask()
                last_option = -1
                last_h = self.policy.get_zero_state()
                flag_done = False
                # store episode stats
                cur_stats = dict(meters=self.task.info['meters'],
                                 target=final_target_name,
                                 options=[])
                #episode_stats['meters'].append()
                #episode_stats['target'].append(final_target_name)
                ep_steps = 0
                ep_reward = 0
                # episode
                for _step in range(planner_steps):
                    planner_input_np = np.concatenate([last_feature,
                                                       accu_mask,
                                                       _target_to_one_hot(last_option),
                                                       final_target])
                    planner_input = torch.from_numpy(planner_input_np).type(FloatTensor)
                    cur_obs.append(planner_input)  # store data
                    # get current option
                    act_ts, nxt_h = self.policy(Variable(planner_input.view(1, 1, -1)), last_h, sample_action=True)  # [batch, seq]
                    last_option = act_ts.data.cpu().numpy().flatten()[0]
                    last_h = nxt_h
                    total_samples += 1
                    cur_stats['options'].append(last_option)
                    # run locomotion
                    motion_data = self.motion.run(combined_target_list[last_option], motion_steps)
                    # process data
                    accu_mask = last_feature
                    for dat in motion_data[:-1]:
                        accu_mask |= dat[0]
                        ep_steps += 1
                        if dat[0][final_target_id] > 0:
                            # TODO: Currently using mask to check whether done! should use "--success-measure"
                            flag_done = True
                            last_feature = dat[0]
                            break
                    cur_info.append((last_option, flag_done, flag_done or (motion_data[-1][0][last_option] > 0)))
                    if flag_done:
                        ep_reward += success_reward
                        break
                    ep_steps += 1
                    ep_reward -= time_penalty
                    last_feature = motion_data[-1][0]
                # update epsiode stats
                cur_stats['steps'] = ep_steps
                cur_stats['good'] = (1 if flag_done else 0)
                cur_stats['reward'] = ep_reward
                episode_stats.append(cur_stats)
                # extra frame for actor-critic
                planner_input_np = np.concatenate([last_feature, accu_mask, _target_to_one_hot(last_option), final_target])
                cur_obs.append(torch.from_numpy(planner_input_np).type(FloatTensor))
                data.append((cur_obs, cur_info))
            # show stats
            _log_it(logger, '  --> Done! Total <%d> Planner Samples Collected! Batch Time Elapsed = %.4fs' % (total_samples, time.time() - tt))
            if (_iter + 1) % eval_rate == 0:
                stats = self._show_stats(episode_stats)
                if (save_dir is not None) and (stats['succ_rate'] > best_eval_rate):
                    _log_it(logger, '  --> Best Succ Model Saved!')
                    best_eval_rate = stats['succ_rate']
                    self.save(save_dir, version='succ')
                eval_stats.append((_iter+1, stats))
            total_episodes += batch_size

            # Perform Training
            _log_it(logger, '> Training ...')
            X, Act, R, Done, Mask = process_batch_data(data, time_penalty, success_reward)
            stats = self._perform_train(X, Act, R, Done, Mask, total_samples)
            train_stats.append(stats)
            _log_it(logger, '  --> Done! Batch Time Elapsed = %.4fs' % (time.time() - tt))
            if (_iter + 1) % report_rate == 0:
                key_lis = sorted(list(stats.keys()))
                for k in key_lis:
                    _log_it(logger, '    >>> %s = %.4f' % (k, stats[k]))
            if (save_dir is not None) and ((_iter + 1) % save_rate == 0):
                self.save(save_dir)
            _log_it(logger, '> Total Time Elasped = %.4fs' % (time.time() - ts))

        self.save(save_dir, version='final')
        dur = time.time() - ts
        _log_it(logger, ("Training Done! Total Computation Time = %.4fs" % dur))
        return train_stats, eval_stats

    def observe(self, exp_data, target):
        for dat in exp_data[:-1]:
            # dat = (mask, act, reward, done)
            self.accu_mask |= dat[0]

    def plan(self, mask, target):
        self.last_mask = mask
        final_target_id = combined_target_index[target]
        curr_feature = np.concatenate([mask,
                                       self.accu_mask,
                                       _target_to_one_hot(self.last_option),
                                       _target_to_one_hot(final_target_id)])
        act_ts, self.last_hidden = self.policy(Variable(curr_feature.view(1, 1, -1)), self.last_hidden, sample_action=True)  # [batch, seq]
        act = act_ts.data.cpu().numpy().flatten()[0]  # option
        self.last_option = act
        self.accu_mask[:] = mask[:]
        return act  # the option

    def reset(self):
        self.last_option = -1
        self.last_mask = None
        self.accu_mask[:] = 0
        self.last_hidden = self._zero_hidden
