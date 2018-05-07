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

n_episode_evaluation = 1000

class ZMQHouseEnvironment:
    def __init__(self, k=0, task_name='roomnav', false_rate=0.0,
                 reward_type='indicator', reward_silence=0,
                 success_measure='see', multi_target=True,
                 include_object_target=True, fixed_target=None, aux_task=False,
                 hardness=None, max_birthplace_steps=None, min_birthplace_grids=0,
                 curriculum_schedule=None,
                 segment_input='none', depth_input=False, target_mask_input=False,
                 cache_supervision=False,
                 include_outdoor_target=True,
                 max_steps=-1, device=0):
        assert k >= 0
        init_birthplace = max_birthplace_steps if curriculum_schedule is None else curriculum_schedule[0]
        self.env = common.create_env(k, task_name=task_name, false_rate=false_rate,
                                     reward_type=reward_type,
                                     hardness=hardness, max_birthplace_steps=init_birthplace,
                                     success_measure=success_measure,
                                     segment_input=segment_input, depth_input=depth_input,
                                     target_mask_input=target_mask_input,
                                     max_steps=max_steps, render_device=device,
                                     genRoomTypeMap=aux_task,
                                     cacheAllTarget=multi_target,
                                     include_object_target=include_object_target,
                                     use_discrete_action=True,   # assume A3C with discrete actions
                                     reward_silence=reward_silence,
                                     #curriculum_schedule=curriculum_schedule,
                                     cache_supervision=cache_supervision,
                                     include_outdoor_target=include_outdoor_target,
                                     min_birthplace_grids=min_birthplace_grids)
        self.obs = self.env.reset() if multi_target else self.env.reset(target='kitchen')
        self.done = False
        self.multi_target = multi_target
        self.fixed_target = fixed_target
        self.aux_task = aux_task
        self.supervision = cache_supervision
        self._sup_act = self.env.info['supervision'] if self.supervision else None
        if self.aux_task:
            #self._aux_target = self.env.get_current_room_pred_mask()   TODO: Currently do not support aux room pred
            assert False, 'Aux Room Prediction Currently Not Supported!'
        self._target = common.target_instruction_dict[self.env.get_current_target()]

    def current_state(self):
        if self.aux_task:
            #return self.obs, self._target, self._aux_target
            return None
        else:
            if not self.supervision:
                return self.obs, self._target
            else:
                return self.obs, self._target, self._sup_act

    def action(self, _act):
        if isinstance(_act, list) or isinstance(_act, tuple):
            act, nxt_birthplace = _act
            self.env.reset_hardness(self.env.hardness, max_birthplace_steps=nxt_birthplace)
        else:
            act = _act
        obs, rew, done, info = self.env.step(act)
        if done:
            if self.multi_target:
                obs = self.env.reset(target=self.fixed_target)
                self._target = common.target_instruction_dict[self.env.get_current_target()]
            else:
                obs = self.env.reset(target=self.env.get_current_target())
            if self.supervision: info = self.env.info
        if self.supervision: self._sup_act = info['supervision']
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
        return ZMQHouseEnvironment(k, config['task_name'], config['false_rate'],
                                   config['reward_type'], config['reward_silence'],
                                   config['success_measure'],
                                   config['multi_target'], config['object_target'],
                                   config['fixed_target'], config['aux_task'],
                                   config['hardness'], config['max_birthplace_steps'],
                                   config['min_birthplace_grids'],
                                   config['curriculum_schedule'],
                                   config['segment_input'], config['depth_input'],
                                   config['target_mask_input'],
                                   (('cache_supervision' in config) and config['cache_supervision']),
                                   config['outdoor_target'],
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
        self.best_succ_rate = 0.0
        self.multi_target = config['multi_target']
        if self.multi_target:
            self.episode_stats['target'] = []
        self.supervision = config['cache_supervision'] if 'cache_supervision' in config else False
        if self.supervision:
            self.curr_sup_act = dict()
        self.aux_task = config['aux_task']
        if self.aux_task:
            self.episode_stats['aux_task_rew'] = []
            self.episode_stats['aux_task_err'] = []
            self.curr_aux_mask = dict()
        self.curriculum_schedule = config['curriculum_schedule']
        self.max_episode_len = config['max_episode_len']
        self.curr_birthplace = dict()
        self.max_birthplace_steps = config['max_birthplace_steps']
        self.global_birthplace = config['max_birthplace_steps'] if self.curriculum_schedule is None else self.curriculum_schedule[0]

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
            action, next_hidden, aux_preds = self.trainer.action(states, hiddens, target=target, return_aux_pred=True)
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
            if (self.curriculum_schedule is not None) and (self.curr_birthplace[id] < self.global_birthplace):
                self.curr_birthplace[id] = self.global_birthplace
                self.send_message(id, (act, self.global_birthplace))  # send action and curriculum
            else:
                self.send_message(id, act)  # send action to simulator
            self.train_buffer[id]['act'].append(act)
            self.hidden_state[id] = next_hidden[i]
            if self.aux_task:
                aux_rew = self.trainer.get_aux_task_reward(int(aux_preds[i]), self.curr_aux_mask[id])
                self.accu_stats[id]['aux_task_rew'] += aux_rew
                self.accu_stats[id]['aux_task_err'] += float(aux_rew < 0)

    def _perform_train(self):
        # prepare training data
        obs = []
        hidden = []
        act = np.zeros((self.batch_size, self.t_max), dtype=np.int32)
        rew = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        done = np.zeros((self.batch_size, self.t_max), dtype=np.float32)
        target = None if not self.multi_target else []
        aux_target = None if not self.aux_task else []
        sup_mask = None if not self.supervision else np.zeros((self.batch_size, self.t_max), dtype=np.uint8)
        for i,id in enumerate(self.train_buffer.keys()):
            dat = self.train_buffer[id]
            obs.append(dat['obs'])
            hidden.append(dat['init_h'])
            act[i] = dat['act']
            if self.supervision:
                # if supervision, change sampled actions to supervised action
                curr_sup_act = np.array(dat['sup_act'][:-1])   # sup_act has t_max+1 elements
                _t_idx = curr_sup_act > -1   # entries with supervision
                act[i, _t_idx] = curr_sup_act[_t_idx]
                sup_mask[i, _t_idx] = 1   # mask those entries with supervision
            rew[i] = dat['rew']
            done[i] = dat['done']
            if target is not None: target.append(dat['target'])
            if self.aux_task: aux_target.append(dat['aux_target'][:-1])
        self.trainer.train()
        if self.aux_task:
            stats = self.trainer.update(obs, hidden, act, rew, done,
                                        target=target, aux_target=aux_target)
        else:
            if self.supervision:
                stats = self.trainer.update(obs, hidden, act, rew, done, target=target, supervision_mask=sup_mask)
            else:
                stats = self.trainer.update(obs, hidden, act, rew, done, target=target)
        if stats is None:
            return False   # just accumulate gradient, no update performed

        self.train_cnt += 1   # update performed!
        # curriculum learning counter increases
        if self.curriculum_schedule is not None:
            if (self.train_cnt % self.curriculum_schedule[2] == 0):
                self.global_birthplace = min(self.max_birthplace_steps, self.global_birthplace + self.curriculum_schedule[1])
        # update stats
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
        return True

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
        self.logger.print("  > Avg Reward = %.6f, Avg Path Len = %.6f, Succ Rate = %.2f, Max-BirthPlace = %d" % (avg_rew, avg_len, avg_succ, self.global_birthplace))
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
        # Best Model with Highest Success Rate
        if avg_succ > self.best_succ_rate:
            self.best_succ_rate = avg_succ
            self.logger.print('   ===========>>>>>>> Best Succ Rate! Model Saved!!!')
            self.trainer.save(self.config['save_dir'], version='succ')
            stats_dict = dict(avg_rew=avg_rew, avg_len=avg_len, avg_succ=avg_succ, iter=self.train_cnt)
            if self.aux_task:
                stats_dict['aux_avg_rew'] = avg_aux_rew
                stats_dict['aux_avg_err'] = avg_aux_err
            self.trainer.save(self.config['log_dir'], version='succ_stats', target_dict_data=stats_dict)
        # Best Model with Highest Avg Reward
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
        sup_act = None
        if self.aux_task:
            state, target, aux_msk = _state
        else:
            if self.supervision:
                state, target, sup_act = _state
            else:
                state, target = _state
            aux_msk = None
        trainer = self.trainer
        if ident not in self.hidden_state:  # new process passed in
            self.hidden_state[ident] = trainer.get_init_hidden()
            self.accu_stats[ident] = dict(rew=0, len=0, succ=0)
            if self.curriculum_schedule is not None:
                self.curr_birthplace[ident] = self.curriculum_schedule[0]
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
            # reset stats
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
        if self.supervision:
            self.curr_sup_act[ident] = sup_act

        # currently run batched simulation
        if ident in self.train_buffer:
            self.train_buffer[ident]['obs'].append(state)
            self.train_buffer[ident]['done'].append(isOver)
            self.train_buffer[ident]['rew'].append(reward)
            if self.multi_target: self.train_buffer[ident]['target'].append(target)
            if self.aux_task: self.train_buffer[ident]['aux_target'].append(trainer.process_aux_target(aux_msk))
            if self.supervision: self.train_buffer[ident]['sup_act'].append(sup_act)
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
                    if self.supervision:
                        self.train_buffer[id]['sup_act'] = [sup_act]
                self._batched_simulate()
                self.pool.clear()
                self.batch_step += 1

        # report stats
        self.comm_cnt += 1
        if self.comm_cnt % self.config['eval_rate'] == 0:
            self._evaluate_stats()
