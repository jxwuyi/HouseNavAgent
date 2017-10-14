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
from zmq_trainer.zmqsimulator import SimulatorProcess, SimulatorMaster, ensure_proc_terminate

n_episode_evaluation = 300

class ZMQHouseEnvironment:
    def __init__(self, k=0, reward_type='indicator', success_measure='center', multi_target=False, aux_task=False,
                 hardness=None, segment_input='none', depth_input=False, max_steps=-1, device=0):
        assert k >= 0
        self.env = common.create_env(k, reward_type=reward_type, hardness=hardness, success_measure=success_measure,
                                     segment_input=segment_input, depth_input=depth_input,
                                     max_steps=max_steps, render_device=device)
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
                                   config['max_episode_len'], device)


class ZMQMaster(SimulatorMaster):
    def __init__(self, pipe1, pipe2, trainer, config):
        super(ZMQMaster, self).__init__(pipe1, pipe2)
        self.config = config
        self.logger = config['logger']
        self.cnt = 0
        self.comm_cnt = 0
        self.train_cnt = 0
        self.trainer = trainer
        self.n_action = common.n_discrete_actions  # TODO: to allow further modification
        self.batch_size = config['batch_size']
        self.t_max = t_max = config['t_max']
        assert t_max > 1, 't_max must be at least 2!'
        self.scheduler = config['scheduler']
        self.train_buffer = dict()
        self.pool = set()
        self.hidden_state = dict()
        self.curr_state = dict()
        self.curr_target = dict()
        self.accu_stats = dict()
        self.batch_step = 0
        self.start_time = time.time()
        self.episode_stats = dict(len=[], rew=[], succ=[])
        self.update_stats = dict(lrate=[])
        self.best_avg_reward = -1e50
        self.multi_target = config['multi_target']
        if self.multi_target:
            self.episode_stats['target'] = []
        self.aux_task = config['aux_task']
        if self.aux_task:
            self.episode_stats['aux_task_rew'] = []
            self.episode_stats['aux_task_err'] = []
            self.curr_aux_mask = dict()

    def _rand_select(self, ids):
        if not isinstance(ids, list): ids = list(ids)
        random.shuffle(ids)
        return ids[:self.batch_size]

    def _batched_simulate(self):
        # random exploration
        batched_ids = list(self.train_buffer.keys())
        states = [[self.curr_state[id]] for id in batched_ids]
        hiddens = [self.hidden_state[id] for id in batched_ids]
        if self.multi_target:
            target = [[self.curr_target[id]] for id in batched_ids]
        else:
            target = None
        self.trainer.eval()  # TODO: check this option
        if self.aux_task:
            action, next_hidden, aux_preds = self.trainer.action(states, hiddens, target=target,
                                                                 compute_aux_pred=True, sample_aux_pred=True)
            aux_preds = aux_preds.squeeze().cpu().numpy()
        else:
            action, next_hidden = self.trainer.action(states, hiddens, target=target)
        cpu_action = action.squeeze().cpu().numpy()
        for i,id in enumerate(batched_ids):
            self.cnt += 1
            if (self.scheduler is not None) and (random.random() > self.scheduler.value(self.cnt)):
                act = random.randint(self.n_action)
            else:
                act = cpu_action[i]
            self.send_message(id, act)  # send action to simulator
            self.train_buffer[id]['act'].append(act)
            self.hidden_state[id] = next_hidden[i]
            if self.aux_task:
                aux_rew = self.trainer.get_aux_task_reward(aux_preds[i], self.curr_aux_mask[id])
                self.accu_stats[id]['aux_task_rew'] += aux_rew
                self.accu_stats[id]['aux_task_err'] += float(aux_rew < 0)

    def _perform_train(self):
        self.train_cnt += 1
        # prepare training data
        obs = []
        hidden = []
        act = np.zeros((self.batch_size, self.t_max), dtype=np.int32)
        rew = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        done = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        target = None if not self.multi_target else []
        for i,id in enumerate(self.train_buffer.keys()):
            dat = self.train_buffer[id]
            obs.append(dat['obs'])
            hidden.append(dat['init_h'])
            act[i] = dat['act']
            rew[i] = dat['rew']
            done[i] = dat['done']
            if target is not None: target.append(dat['target'])
        self.trainer.train()
        stats = self.trainer.update(obs, hidden, act, rew, done, target=target)
        for key in stats.keys():
            if key not in self.update_stats:
                self.update_stats[key] = []
            self.update_stats[key].append(stats[key])
        if 'lrate' not in stats:
            self.update_stats['lrate'].append(self.trainer.lrate)
        if self.train_cnt % self.config['report_rate'] == 0:
            self.logger.print('Training Iter#%d ...' % self.train_cnt)
            keys = sorted(stats.keys())
            for key in keys:
                if isinstance(stats[key], str):
                    self.logger.print('%s = %s' % (key, stats[key]))
                else:
                    self.logger.print('  >>> %s = %.5f' % (key, stats[key]))
        if self.train_cnt % self.config['save_rate'] == 0:
            self.trainer.save(self.config['save_dir'])
        if self.train_cnt > self.config['max_iters']:
            print('Finished All the Iters!!!')
            self._evaluate_stats()
            self.save_all(version='final')
            exit(0)

    def save_all(self, version=''):
        self.trainer.save(self.config['save_dir'], version=version)
        self.trainer.save(self.config['log_dir'], version=version+'_epis_stats', target_dict_data=self.episode_stats)
        self.trainer.save(self.config['log_dir'], version=version+'_update_stats', target_dict_data=self.update_stats)

    # TODO: TO handle Aux_Task [to output the accumulative reward and error for aux task]
    def _evaluate_stats(self):
        duration = time.time() - self.start_time
        self.logger.print("+++++++++++++++++++ Eval +++++++++++++++++++++++++++++++")
        self.logger.print("Running Stats <#Samles = {}>".format(self.comm_cnt))
        self.logger.print("> Time Elapsed = %.4f min"%(duration / 60))
        self.logger.print(" -> #Episode = {}, #Updates = {}".format(len(self.episode_stats['rew']), self.train_cnt))
        rew_stats = self.episode_stats['rew'][-n_episode_evaluation:]
        len_stats = self.episode_stats['len'][-n_episode_evaluation:]
        succ_stats = self.episode_stats['succ'][-n_episode_evaluation:]
        if self.multi_target:
            tar_stats = self.episode_stats['target'][-n_episode_evaluation:]
        avg_rew = sum(rew_stats) / len(rew_stats)
        avg_len = sum(len_stats) / len(len_stats)
        avg_succ = sum(succ_stats) / len(succ_stats)
        self.logger.print("  > Avg Reward = %.6f, Avg Path Len = %.6f, Succ Rate = %.2f" % (avg_rew, avg_len, avg_succ))
        if self.aux_task:
            aux_rew_stats = self.episode_stats['aux_task_rew'][-n_episode_evaluation:]
            aux_err_stats = self.episode_stats['aux_task_err'][-n_episode_evaluation:]
            avg_aux_rew = sum(aux_rew_stats) / len(aux_rew_stats)
            avg_aux_err = sum(aux_err_stats) / len(aux_err_stats)
            self.logger.print(
                "  ---> Aux-Task Predictions: Avg Rew = %.3f,  Avg Err = %.3f" % (avg_aux_rew, avg_aux_err))
        if self.multi_target:
            all_stats = dict()
            for i, t in enumerate(tar_stats):
                if t not in all_stats:
                    all_stats[t] = [0.0, 0.0, 0.0, 0.0]
                all_stats[t][0] += 1
                all_stats[t][1] += rew_stats[i]
                all_stats[t][2] += len_stats[i]
                all_stats[t][3] += succ_stats[i]
            m = len(rew_stats)
            for t in all_stats.keys():
                n, r, l, s = all_stats[t]
                self.logger.print("  ---> Mul-Target <%s> Rate = %.3f, Avg Rew = %.3f, Avg Len = %.3f, Succ Rate = %.3f"
                                  % (common.all_target_instructions[t], n/m, r/n, l/n, s/n))
        self.logger.print("  >>>> Total FPS: %.5f"%(self.comm_cnt * 1.0 / duration))
        self.logger.print('   ----> Data Loading Time = %.4f min' % (time_counter[0] / 60))
        self.logger.print('   ----> Training Time = %.4f min' % (time_counter[1] / 60))
        self.logger.print('   ----> Update Time Per Iter = %.4f s' % (duration / self.train_cnt))
        if avg_rew > self.best_avg_reward:
            self.best_avg_reward = avg_rew
            self.logger.print('   ===========>>>>>>> Best Avg Reward! Model Saved!!!')
            self.trainer.save(self.config['save_dir'], version='best')
            stats_dict = dict(avg_rew=avg_rew, avg_len=avg_len, avg_succ=avg_succ, iter=self.train_cnt)
            if self.aux_task:
                stats_dict['aux_avg_rew'] = avg_aux_rew
                stats_dict['aux_avg_err'] = avg_aux_err
            self.trainer.save(self.config['log_dir'], version='best_stats', target_dict_data=stats_dict)
        self.logger.print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

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
            self.hidden_state[ident] = trainer.get_init_hidden()
            self.accu_stats[ident] = dict(rew=0, len=0, succ=0)
            if self.multi_target:
                self.accu_stats[ident]['target'] = target
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
            if self.multi_target:
                self.episode_stats['target'].append(self.accu_stats[ident]['target'])
                self.accu_stats[ident]['target'] = target
            if self.aux_task:
                self.episode_stats['aux_task_err'].append(self.accu_stats[ident]['aux_task_err'] / self.episode_stats['len'][-1])
                self.accu_stats[ident]['aux_task_err'] = 0
                self.episode_stats['aux_task_rew'].append(self.accu_stats[ident]['aux_task_rew'])
                self.accu_stats[ident]['aux_task_rew'] = 0

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
