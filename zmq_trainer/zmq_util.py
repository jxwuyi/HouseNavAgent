from headers import *
import threading
import numpy as np
import random
import sys
import time
import os
import json
import utils
import common
from zmqsimulator import SimulatorProcess, SimulatorMaster, ensure_proc_terminate

class ZMQHouseEnvironment:
    def __init__(self, k=0, hardness=None, segment_input='none', depth_input=False, max_steps=-1, device=0):
        assert k >= 0
        self.env = common.create_env(k, hardness=hardness, segment_input=segment_input, depth_input=depth_input,
                                     max_steps=max_steps, render_device=device)
        self.obs = self.env.reset()
        self.done = False

    def current_state(self):
        return self.obs

    def action(self, act):
        obs, rew, done, _ = self.env.step(act, return_info=False)
        if done:
            obs = self.env.reset()
        self.obs = obs
        return rew, done

class ZMQSimulator(SimulatorProcess):
    def _build_player(self):
        config = self.config
        k = self.idx % config['n_house']
        device_list = config['render_devices']
        device_ind = self.idx % len(device_list)
        device = device_list[device_ind]
        return ZMQHouseEnvironment(k, True, config['hardness'],
                                    config['segment_input'], config['depth_input'], device)


class ZMQMaster(SimulatorMaster):
    def __init__(self, pipe1, pipe2, trainer, scheduler, batch_size, t_max, config):
        super(ZMQMaster, self).__init__(pipe1, pipe2)
        self.config = config
        self.logger = utils.MyLogger(config['log_dir'], True)
        self.cnt = 0
        self.comm_cnt = 0
        self.train_cnt = 0
        self.trainer = trainer
        self.n_action = common.n_discrete_actions  # TODO: to allow further modification
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.t_max = t_max
        assert t_max > 1, 't_max must be at least 2!'
        self.train_buffer = dict()
        self.pool = set()
        self.hidden_state = dict()
        self.curr_state = dict()
        self.accu_stats = dict()
        self.batch_step = 0
        self.start_time = time.time()
        self.episode_stats = dict(len=[], rew=[])

    def _rand_select(self, ids):
        random.shuffle(ids)
        return ids[:self.batch_size]

    # TODO: to test whether to use batched_simulation or async-single-proc simulation!
    def _batched_simulate(self):
        # random exploration
        states = [[self.curr_state[id]] for id in self.curr_state.keys()]
        hiddens = [self.hidden_state[id] for id in self.curr_state.keys()]
        self.trainer.eval()  # TODO: check this option
        action, next_hidden = self.trainer.action(states, hiddens)
        for i,id in enumerate(self.curr_state.keys()):
            self.cnt += 1
            if random.random() > self.scheduler.value(self.cnt):
                act = random.randint(self.n_action)

            else:
                act = action[i]
            self.send_message(id, act)  # send action to simulator
            self.train_buffer[id]['act'].append(act)
            self.hidden_state[id] = next_hidden[i]

    def _perform_train(self):
        self.train_cnt += 1
        # prepare training data
        obs = []
        hidden = []
        act = np.zeros((self.batch_size, self.t_max), dtype=np.int32)
        rew = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        done = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        for i,id in enumerate(self.train_buffer.keys()):
            dat = self.train_buffer[id]
            obs.append(dat['obs'])
            hidden.append(dat['init_h'])
            act[i] = dat['act']
            rew[i] = dat['rew']
            done[i] = dat['done']
        self.trainer.train()
        stats = self.trainer.update(obs, hidden, act, rew, done)
        if self.train_cnt % self.config['train_log_rate'] == 0:
            self.logger.print('Training Iter#%d ...' % self.train_cnt)
            keys = sorted(stats.keys())
            for key in keys:
                self.logger.print('  >>> %s = %.5f' % (key, stats[key]))
        if self.train_cnt % self.config['save_rate'] == 0:
            self.trainer.save(self.config['save_dir'])

    def recv_message(self, ident, state, reward, isOver):
        """
        Handle a message sent by "ident" simulator.
        The simulator takes the last action we send, arrive in "state", get "reward" and "isOver".
        """
        trainer = self.trainer
        if ident not in self.hidden_state:  # new process passed in
            self.hidden_state[ident] = trainer.get_init_hidden()
            self.accu_stats[ident] = dict(rew=0, len=0)

        self.accu_stats[ident]['rew'] += reward
        self.accu_stats[ident]['len'] += 1

        if isOver:
            self.hidden_state[ident] *= 0.0
            # accumulate running stats
            self.episode_stats['rew'].append(self.accu_stats[ident]['rew'])
            self.episode_stats['len'].append(self.accu_stats[ident]['len'])
            self.accu_stats[ident]['rew'] = 0
            self.accu_stats[ident]['len'] = 1

        if isinstance(state, np.ndarray): state = torch.from_numpy(state).type(ByteTensor)
        self.curr_state[ident] = state

        # currently run batched simulation
        if ident in self.train_buffer:
            self.train_buffer[ident]['obs'].append(state)
            self.train_buffer[ident]['done'].append(isOver)
            self.train_buffer[ident]['rew'].append(reward)
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
                                                 init_h=self.hidden_state[id])
                self._batched_simulate()
                self.pool.clear()
                self.batch_step += 1

        # report stats
        if self.comm_cnt % self.config['eval_rate'] == 0:
            duration = time.time() - self.start_time
            self.logger.print("Running Stats <#Samles = {}>".format(self.comm_cnt))
            self.logger.print("> Time Elapsed = %.4f min".format(duration/60))
            self.logger.print(" -> #Episode = {}, #Updates = {}".format(len(self.episode_stats), self.train_cnt))
            rew_stats = self.episode_stats['rew'][-500:]
            len_stats = self.episode_stats['len'][-500:]
            self.logger.print("  > Avg Reward = %.6f, Avg Path Len = %.6f" % (sum(rew_stats)/len(rew_stats), sum(len_stats)/len(len_stats)))
            self.logger.print("  >>>> Total FPS: ", self.comm_cnt * 1.0 / duration)
            self.logger.print('   ----> Data Loading Time = %.4f min' % (time_counter[0] / 60))
            self.logger.print('   ----> Training Time = %.4f min' % (time_counter[1] / 60))
