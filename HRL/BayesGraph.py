from headers import *

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

"""
Learning a Bayesian Graph over Objects and Rooms
Rooms: 9 rooms total
    8 rooms + indoor
Objects: 15 objects total
Parameters:
  --> for each pair of rooms R1, R2: connect(R1,R2) ~ Bernoulli(theta(R1, R2))
  --> for each room R and object O: contain(R, O) ~ Bernoulli(theta(R, O))
  --> noisy observation parameters: if connect(X, Y) == 1: obs ~ Bernoulli(psi_1)
                                    else: obs ~ Bernoulli(psi_2)
Total #Params = 9 * 8 / 2 + 9 * 15 + 2 = 308
"""

combined_target_list = ALLOWED_TARGET_ROOM_TYPES + ALLOWED_OBJECT_TARGET_TYPES
combined_target_index = dict()
for i, t in enumerate(combined_target_list):
    combined_target_index[t] = i
independent_object_index = dict()
for i, o in enumerate(ALLOWED_OBJECT_TARGET_TYPES):
    independent_object_index[o] = i

n_mask_feature = len(ALLOWED_OBJECT_TARGET_TYPES) + len(ALLOWED_TARGET_ROOM_TYPES)

all_graph_rooms_names = ALLOWED_TARGET_ROOM_TYPES + ['indoor']
n_rooms = len(ALLOWED_TARGET_ROOM_TYPES) + 1   # room types + indoor
n_objects = len(ALLOWED_OBJECT_TARGET_TYPES)


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


class GraphPlanner(object):
    def __init__(self, motion, task=None):
        self.task = motion.task if motion is not None else task
        self.env = self.task.env
        self.motion = motion
        self.n_param = n_rooms * (n_rooms - 1) // 2 + n_rooms * n_objects + 2
        self.conn_rooms = np.ones((n_rooms, n_rooms), dtype=np.float32)
        self.conn_objs = np.ones((n_rooms, n_objects), dtype=np.float32) * 0.1
        self.conn_noise = np.array([0.001, 0.95], dtype=np.float32)
        self.params = [self.conn_rooms, self.conn_objs, self.conn_noise]
        self.param_idx = [(0, (i, j)) for i in range(n_rooms) for j in range(i+1, n_rooms)] + \
                         [(1, (i, j)) for i in range(n_rooms) for j in range(n_objects)] + \
                         [(2, 0), (2, 1)]
        self.param_size = n_rooms * (n_rooms - 1) // 2 + n_rooms * n_objects + 2
        self.param_group_size = [n_rooms * (n_rooms - 1) // 2, n_rooms * n_objects, 2]
        self.g_rooms = np.zeros((n_rooms, n_rooms), dtype=np.float32)
        self.g_objs = np.zeros((n_rooms, n_objects), dtype=np.float32)
        self.obs_rooms = np.zeros((n_rooms, n_rooms), dtype=np.int32)
        self.obs_objs = np.zeros((n_rooms, n_objects), dtype=np.int32)
        self.exp_rooms = np.zeros((n_rooms, n_rooms), dtype=np.int32)
        self.exp_objs = np.zeros((n_rooms, n_objects), dtype=np.int32)
        self.stats_exp = [self.exp_rooms, self.exp_objs]
        self.stats_obs = [self.obs_rooms, self.obs_objs]
        self.graph = [self.g_rooms, self.g_objs]
        self.cached_eps = 0
        self.excluded_targets = set()
        self.excluded_targets.add('indoor')

    def add_excluded_target(self, target):
        self.excluded_targets.add(target)

    def get_target_index(self, target):
        if target in combined_target_index:
            return combined_target_index[target]
        else:
            return -1

    def get_param_size(self):
        return self.param_size

    def get_param_group_size(self):
        return self.param_group_size

    def get_param(self, i):
        i = (i % self.param_size)
        idx = self.param_idx[i]
        return self.params[idx[0]][idx[1]]

    def set_param(self, i, val):
        i = (i % self.param_size)
        idx = self.param_idx[i]
        if idx[0] < 2:
            x, y = idx[1]
            self.params[idx[0]][x,y] = self.params[idx[0]][y, x] = val
        else:
            self.params[idx[0]][idx[1]] = val

    def parameters(self):
        return self.params

    def set_parameters(self, params):
        assert len(params) == 3
        assert params[0].shape == (n_rooms, n_rooms)
        assert params[1].shape == (n_rooms, n_objects)
        assert params[2].shape == (2,)
        self.conn_rooms = params[0]
        self.conn_objs = params[1]
        self.conn_noise = params[2]
        self.params = params
        self.cached_eps = max(1e-4, np.min(self.params[0]))

    """
    n_trials: the number of episodes to estimate the connectivity between two rooms
    max_allowed_steps: the number of steps allowed for the exploration
    NOTE:
        -- total number of episodes with <n_trial> per room:
           > n_trial * n_room (8) * (n_room-1) = 56 * n_trials
        -- total steps will be 56 * n_trials * max_allowed_steps
        -- sum up: 56 * n_trials * max_allowed_steps * #houses
        -- default params on 200 houses -> 8,400,000 frames
    """
    def learn(self, n_trial=25, max_allowed_steps=30, eps=1e-4, logger=None):
        self.cached_eps = eps
        if hasattr(self.env, 'all_houses'):
            all_houses = self.env.all_houses
        else:
            all_houses = [self.env.house]

        # set hardness of the task to 0
        self.task.reset_hardness(hardness=0)
        # learning graph prior over rooms
        cnt_rooms = np.zeros((n_rooms, n_rooms), dtype=np.int32)
        cnt_objs = np.zeros((n_rooms, n_objects), dtype=np.int32)
        pos_objs = np.zeros((n_rooms, n_objects), dtype=np.int32)
        pos_rooms = np.zeros((n_rooms, n_rooms), dtype=np.int32)

        ts = time.time()
        _log_it(logger, "House Enumerating & Sampling ... Total {} Houses...".format(len(all_houses)))
        for _i, house in enumerate(all_houses):
            _log_it(logger, ">> House#{} ...".format(_i))
            self.env.reset_house(house._id)
            all_rooms = house.all_desired_roomTypes
            all_objects = house.all_desired_targetObj
            # check connectivity to indoor
            indoor_id = n_rooms - 1
            in_msk, out_msk = house.getRegionMaskForRoomMask(0)
            if in_msk is not None: # has region indoor
                for r in all_rooms:   # connect to other rooms
                    r_id = combined_target_index[r]
                    cnt_rooms[r_id,indoor_id] += 1
                    cnt_rooms[indoor_id, r_id] += 1
                    if out_msk[r_id] > 0:
                        pos_rooms[r_id, indoor_id] += 1
                        pos_rooms[indoor_id, r_id] += 1
                for o in all_objects:  # connect to objects
                    o_id = independent_object_index[o]
                    o_pos = combined_target_index[o]
                    cnt_objs[indoor_id, o_id] += 1
                    if in_msk[o_pos] > 0:
                        pos_objs[indoor_id, o_id] += 1
            # check other normal room types
            for r1 in all_rooms:
                r1_id = combined_target_index[r1]
                in_msk, out_msk = house.getRegionMaskForTarget(r1)
                # connectivity towards other rooms
                for r2 in all_rooms:
                    if r1 == r2: continue
                    r2_id = combined_target_index[r2]
                    if out_msk[r2_id] > 0:
                        pos_rooms[r1_id, r2_id] += 1
                        cnt_rooms[r1_id, r2_id] += 1
                    else:  # not connected closely, need to run exploration
                        cnt_rooms[r1_id, r2_id] += n_trial
                        n_pos = 0
                        for _ in range(n_trial):
                            cx, cy = house.getRandomLocation(r1)
                            self.task.reset(target=r2, reset_house=False, birthplace=(cx, cy))
                            D = self.motion.run(r2, max_allowed_steps)
                            if D[-1][3] or any([(d[0][r2_id] > 0) for d in D]):
                                n_pos += 1
                        pos_rooms[r1_id, r2_id] += n_pos
                # connecitivity towards objects
                for o in all_objects:
                    o_id = independent_object_index[o]
                    o_pos = combined_target_index[o]
                    cnt_objs[r1_id, o_id] += 1
                    if in_msk[o_pos] > 0:
                        pos_objs[r1_id, o_id] += 1

            _log_it(logger, "  ---> %d / %d houses processed! time elapsed = %.4fs" % (_i+1, len(all_houses), time.time()-ts))
        dur = time.time() - ts
        _log_it(logger, ("Sampling Done! Total Sampling Time Elapsed = %.4fs" % dur))

        # compute all the MLE for graph parameters
        _log_it(logger, "Computing Statistics and Parameters ...")
        for r1 in range(n_rooms):
            for r2 in range(r1+1, n_rooms):
                base_cnt = cnt_rooms[r1, r2] + cnt_rooms[r2, r1]
                pos_cnt = pos_rooms[r1, r2] + pos_rooms[r2, r1]
                if base_cnt == 0:  # no positive connectivity at all
                    self.conn_rooms[r1, r2] = self.conn_rooms[r2, r1] = eps
                else:
                    self.conn_rooms[r1, r2] = self.conn_rooms[r2, r1] = pos_cnt / base_cnt
            for o in range(n_objects):
                if cnt_objs[r1, o] == 0:
                    self.conn_objs[r1, o] = 0
                else:
                    self.conn_objs[r1, o] = pos_objs[r1, o] / cnt_objs[r1, o]
        dur = time.time() - ts
        _log_it(logger, ("Training Done! Total Computation Time = %.4fs" % dur))

    def evolve(self):
        raise NotImplementedError()

    """
    Variable X ~ Bernoulli(p)
    Get Y = <n_total> noisy samples from X, <n_pos> are 1, <n_neg> are 0
    compute Pr[X = 1 | Y] = Pr[X=1, Y] / (Pr[X=1, Y] + Pr[X=0, Y]) = W_1 / (W_0 + W_1)
    """
    def _update_graph(self, visit):
        for entry in visit:
            t, x, y = entry
            n_total = self.stats_exp[t][x, y]
            n_pos = self.stats_obs[t][x, y]
            if t == 0:
                n_total += self.stats_exp[t][y, x]
                n_pos += self.stats_obs[t][y, x]
            n_neg = n_total - n_pos
            p = self.params[t][x, y]
            # compute Pr[X=0, Y]
            lg_W_0 = np.log(max(1 - p, 1e-10)) + np.log(1 - self.params[2][0]) * n_neg + np.log(self.params[2][0]) * n_pos
            # compute Pr[X=1, Y]
            lg_W_1 = np.log(max(p, 1e-10)) + np.log(1 - self.params[2][1]) * n_neg + np.log(self.params[2][1]) * n_pos
            max_lg = max(lg_W_0, lg_W_1)
            lg_W_0 -= max_lg
            lg_W_1 -= max_lg
            W_0 = np.exp(lg_W_0)
            W_1 = np.exp(lg_W_1)
            self.graph[t][x, y] = W_1 / (W_1 + W_0)
            if t == 0:
                self.graph[t][y, x] = self.graph[t][x, y]

    def observe(self, exp_data, target):
        # execute sub-policy <target>, observe experiences <data>
        orig_mask = prev_mask = exp_data[0][0]
        full_mask = orig_mask.copy()
        visit = set()
        for i, dat in enumerate(exp_data):
            msk = dat[0]  # dat = (mask, act, reward, done)
            full_mask |= msk
            if any(msk != prev_mask):
                curr_rooms = _get_room_index_from_mask(msk)
                curr_objs = _get_object_index_from_mask(msk)
                for r in curr_rooms:
                    for o in curr_objs:   #  add visited objects
                        visit.add((1, r, o))
                        self.exp_objs[r, o] += 1
                        self.obs_objs[r, o] += 1
                prev_mask = msk

        target_id = combined_target_index[target]
        if full_mask[target_id] == 0:   # fail to reach target!!!!!
            orig_rooms = _get_room_index_from_mask(orig_mask)
            if target in ALLOWED_PREDICTION_ROOM_TYPES:   # target is a room
                for r in orig_rooms:
                    visit.add((0, r, target_id))
                    self.exp_rooms[r, target_id] += 1
            else:  # target is an object
                object_id = independent_object_index[target]
                for r in orig_rooms:
                    visit.add((1, r, object_id))
                    self.exp_objs[r, object_id] += 1

        # add all pairs of connected rooms
        full_rooms = _get_room_index_from_mask(full_mask)
        for i in range(len(full_rooms)):
            for j in range(i + 1, len(full_rooms)):
                visit.add((0, full_rooms[i], full_rooms[j]))
                self.exp_rooms[i, j] += 1
                self.obs_rooms[i, j] += 1

        # compute posterior
        self._update_graph(visit)

    def plan(self, mask, target, return_list = False):
        # find shortest path towards target in self.graph
        target_id = combined_target_index[target]
        if mask[target_id] > 0:
            return target if not return_list else [target]  # already there, directly go

        #print('[Graph] current house id = {}'.format(self.task.house._id))
        #print('[Graph] start planning ... target = {}, id = {}, mask = {}'.format(target, target_id, mask))

        # shortest path planning
        curr_rooms = _get_room_index_from_mask(mask)
        opt_rooms = np.zeros(n_rooms, dtype=np.float32)
        prev_rooms = np.ones(n_rooms, dtype=np.int32) * -1
        for r in curr_rooms:
            opt_rooms[r] = 1
        visit = set()
        # shorest path over rooms
        for _ in range(n_rooms):
            sl_r = -1
            for i in range(n_rooms):
                if i not in visit:
                    if (sl_r < 0) or (opt_rooms[i] > opt_rooms[sl_r]): sl_r = i
            if sl_r < 0:
                break
            visit.add(sl_r)
            for i in range(n_rooms):
                if i in visit: continue
                cur_p = opt_rooms[sl_r] * self.g_rooms[sl_r, i]
                if cur_p > opt_rooms[i]:
                    opt_rooms[i] = cur_p
                    prev_rooms[i] = sl_r
        #print('[Graph] SSP Done!')
        full_plan = []
        if target in ALLOWED_OBJECT_TARGET_INDEX:   # object target
            full_plan.append(target)
            object_id = independent_object_index[target]
            tar_room = -1
            best_p = 0
            for r in range(n_rooms - 1):   # exclude <indoor>
                if all_graph_rooms_names[r] in self.excluded_targets: continue
                curr_p = opt_rooms[r] * self.g_objs[r, object_id]
                if (tar_room < 0) or (best_p < curr_p):
                    best_p = curr_p
                    tar_room = r
        else:  #  room target
            tar_room = target_id
        # need to reach room <tar_room>
        #print('[Graph] curr_rooms = {}'.format(curr_rooms))
        #print('[Graph] opt_rooms = {}'.format(opt_rooms))
        #print('[Graph] prev_rooms = {}'.format(prev_rooms))
        #print('[Graph] Target Room ID = {}, Type = {}, Probability = {}'.format(tar_room, all_graph_rooms_names[tar_room], opt_rooms[tar_room]))
        ptr = tar_room
        if ptr in curr_rooms: # we should directly execute target
            return target if not return_list else [target]
        ptr_name = all_graph_rooms_names[ptr]
        if ptr_name not in self.excluded_targets:
            full_plan.append(ptr_name)
        assert prev_rooms[ptr] > -1, '[BayesGraph.plan] Currently Target Room is {}, however it is not reachable!!!! curr house id = {}'.format(combined_target_list[ptr], self.task.house._id)
        #print('[Graph] Fetching SSP Path ...')
        while prev_rooms[ptr] not in curr_rooms:
            ptr = prev_rooms[ptr]
            ptr_name = all_graph_rooms_names[ptr]
            if ptr_name not in self.excluded_targets:
                full_plan.append(ptr_name)
            #print('[Graph] --> add ptr = {}, prev = {}'.format(all_graph_rooms_names[ptr], prev_rooms[ptr]))
            assert prev_rooms[ptr] > -1
        full_plan.reverse()
        #print('[Graph] Done! Path = {}'.format(full_plan))
        return full_plan[0] if not return_list else full_plan

    def reset(self):
        self.g_rooms[...] = self.conn_rooms
        self.g_objs[...] = self.conn_objs
        self.obs_rooms[...] = 0
        self.obs_objs[...] = 0
        self.exp_rooms[...] = 0
        self.exp_objs[...] = 0

    #########################
    # DEBUG Functionalities
    #########################
    def _show_prior_object(self, object_id=None, logger=None):
        object_range = list(range(n_objects)) if object_id is None else [object_id]
        for i in object_range:
            _log_it(logger, 'Object#{}, <{}>:'.format(i, ALLOWED_OBJECT_TARGET_TYPES[i]))
            for r in range(n_rooms):
                r_name = 'indoor' if r == n_rooms - 1 else ALLOWED_TARGET_ROOM_TYPES[r]
                if self.conn_objs[r, i] > self.cached_eps + 1e-9:
                    _log_it(logger, '  --> Room#{}, <{}>, Prob = {}'.format(r, r_name, self.conn_objs[r, i]))


    def _show_prior_room(self, room_id=None, logger=None):
        room_range = list(range(n_rooms)) if room_id is None else [room_id]
        for r in room_range:
            r_name = 'indoor' if r == n_rooms - 1 else ALLOWED_TARGET_ROOM_TYPES[r]
            _log_it(logger, 'Room#{}, <{}>:'.format(r, r_name))
            for y in range(n_rooms):
                if y == r: continue
                y_name = 'indoor' if y == n_rooms - 1 else ALLOWED_TARGET_ROOM_TYPES[y]
                if self.conn_rooms[r, y] > self.cached_eps + 1e-9:
                    _log_it(logger, '  --> Room#{}, <{}>, Prob = {}'.format(y, y_name, self.conn_rooms[r, y]))

            for o in range(n_objects):
                o_name = ALLOWED_OBJECT_TARGET_TYPES[o]
                if self.conn_objs[r, o] > self.cached_eps + 1e-9:
                    _log_it(logger, '  --> Object#{}, <{}>, Prob = {}'.format(o, o_name, self.conn_objs[r, o]))

    def _show_posterior_room(self, room_id=None, logger=None):
        room_range = list(range(n_rooms)) if room_id is None else [room_id]
        for r in room_range:
            r_name = 'indoor' if r == n_rooms - 1 else ALLOWED_TARGET_ROOM_TYPES[r]
            _log_it(logger, 'Room#{}, <{}>:'.format(r, r_name))
            for y in range(n_rooms):
                if y == r: continue
                y_name = 'indoor' if y == n_rooms - 1 else ALLOWED_TARGET_ROOM_TYPES[y]
                if self.conn_rooms[r, y] > self.cached_eps + 1e-9:
                    _log_it(logger, '  --> Room#%d, <%s>, Prob = %.5f (prior = %.5f)' % (y, y_name, self.g_rooms[r, y], self.conn_rooms[r, y]))

            for o in range(n_objects):
                o_name = ALLOWED_OBJECT_TARGET_TYPES[o]
                if self.conn_objs[r, o] > self.cached_eps + 1e-9:
                    _log_it(logger, '  --> Object#%d, <%s>, Prob = %.5f (prior = %.5f)' % (o, o_name, self.g_objs[r, o], self.conn_objs[r, o]))
